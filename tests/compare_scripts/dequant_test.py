from pathlib import Path
from typing import Optional, Sequence

import torch
import triton
from triton import language as tl

import g4b.device
from g4b import tensor
from g4b.gguf import GGUFType, GGUFTensor
from g4b.kernels.matmul import matmul_a3d_b2d_b_loader_jfn
from g4b.kernels.utils import launch
from scripts.reference_impl import dequant_q4k_to_fp32, dequant_q5k_to_fp32, dequant_q6k_to_fp32


g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()


@triton.jit
def _dequant_loader_test_kernel(
    out_ptr,
    q_ptr,
    out_shape0: tl.constexpr,
    out_shape1: tl.constexpr,
    q_shape0: tl.constexpr,
    q_shape1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    q_stride0: tl.constexpr,
    q_stride1: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    CONCEPTUAL_DTYPE: tl.constexpr,
):
    row_off = tl.program_id(1) * BLOCK_ROWS
    col_off = tl.program_id(0) * BLOCK_COLS

    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        (1, q_shape0, q_shape1),
        (0, q_stride0, q_stride1),
        (1, BLOCK_ROWS, BLOCK_COLS),
    )

    tile = matmul_a3d_b2d_b_loader_jfn(
        "q",
        q_desc,
        0,
        row_off,
        col_off,
        q_ptr,
        1,
        q_shape0,
        q_shape1,
        0,
        q_stride0,
        q_stride1,
        1,
        BLOCK_ROWS,
        BLOCK_COLS,
        CONCEPTUAL_DTYPE,
    )

    out_row = row_off + tl.arange(0, BLOCK_ROWS)[None, :, None]
    out_col = col_off + tl.arange(0, BLOCK_COLS)[None, None, :]
    mask = (out_row < out_shape0) & (out_col < out_shape1)
    tl.store(out_ptr + out_row * out_stride0 + out_col * out_stride1, tile, mask=mask)


def make_random_q4k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 144), dtype=torch.uint8)

    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    md = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 0:2] = dd
    raw[:, 2:4] = md
    raw[:, 4:16] = torch.randint(0, 256, (n_blocks, 12), dtype=torch.uint8)
    raw[:, 16:] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)

    return GGUFTensor(
        name="synthetic.q4_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q4_K,
        data=raw.numpy().tobytes(),
    )


def make_random_q4k_test_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    return make_random_q4k_tensor(rows, cols), {}


def make_random_q5k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 176), dtype=torch.uint8)

    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    md = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 0:2] = dd
    raw[:, 2:4] = md
    raw[:, 4:16] = torch.randint(0, 256, (n_blocks, 12), dtype=torch.uint8)
    raw[:, 16:48] = torch.randint(0, 256, (n_blocks, 32), dtype=torch.uint8)
    raw[:, 48:] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)

    return GGUFTensor(
        name="synthetic.q5_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q5_K,
        data=raw.numpy().tobytes(),
    )


def make_random_q5k_test_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    return make_random_q5k_tensor(rows, cols), {}


def make_random_q6k_tensor(rows: int, cols: int) -> GGUFTensor:
    assert cols % 256 == 0
    n_blocks = rows * (cols // 256)
    raw = torch.empty((n_blocks, 210), dtype=torch.uint8)

    raw[:, 0:128] = torch.randint(0, 256, (n_blocks, 128), dtype=torch.uint8)
    raw[:, 128:192] = torch.randint(0, 256, (n_blocks, 64), dtype=torch.uint8)
    raw[:, 192:208] = torch.randint(-128, 128, (n_blocks, 16), dtype=torch.int8).view(torch.uint8)
    dd = (torch.rand((n_blocks,), dtype=torch.float16) * 0.25 + 0.01).view(torch.uint8).reshape(n_blocks, 2)
    raw[:, 208:210] = dd

    return GGUFTensor(
        name="synthetic.q6_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q6_K,
        data=raw.numpy().tobytes(),
    )


def make_random_q6k_test_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    return make_random_q6k_tensor(rows, cols), {}


def _pack_q4k_scales_and_mins(sc: torch.Tensor, mins: torch.Tensor) -> torch.Tensor:
    assert sc.shape == mins.shape
    assert sc.shape[-1] == 8
    assert sc.dtype == torch.uint8
    assert mins.dtype == torch.uint8
    assert torch.all(sc < 64)
    assert torch.all(mins < 64)

    n_blocks = sc.shape[0]
    packed = torch.empty((n_blocks, 12), dtype=torch.uint8)
    d_frags = packed[:, 0:4]
    m_frags = packed[:, 4:8]
    mixed_frags = packed[:, 8:12]

    d_frags[:] = (sc[:, 0:4] & 0x3F) | ((sc[:, 4:8] & 0x30) << 2)
    m_frags[:] = (mins[:, 0:4] & 0x3F) | ((mins[:, 4:8] & 0x30) << 2)
    mixed_frags[:] = (sc[:, 4:8] & 0x0F) | ((mins[:, 4:8] & 0x0F) << 4)
    return packed


def _pack_q4k_qs(qs: torch.Tensor) -> torch.Tensor:
    assert qs.ndim == 3
    assert qs.shape[1:] == (8, 32)
    assert qs.dtype == torch.uint8
    assert torch.all(qs < 16)

    low = qs[:, 0::2, :]
    high = qs[:, 1::2, :]
    return (low | (high << 4)).reshape((qs.shape[0], 128))


def _pack_q5k_qh(qs: torch.Tensor) -> torch.Tensor:
    assert qs.ndim == 3
    assert qs.shape[1:] == (8, 32)
    assert qs.dtype == torch.uint8
    assert torch.all(qs < 32)

    high_bits = (qs >> 4) & 0x01
    bit_shifts = torch.arange(8, dtype=torch.uint8).reshape((1, 8, 1))
    return ((high_bits << bit_shifts).sum(dim=1) & 0xFF).to(torch.uint8)


def _pack_q6k_ql(qs: torch.Tensor) -> torch.Tensor:
    assert qs.ndim == 3
    assert qs.shape[1:] == (16, 16)
    assert qs.dtype == torch.uint8
    assert torch.all(qs < 64)

    low = (qs & 0x0F).reshape((qs.shape[0], 2, 2, 64))
    return (low[:, :, 0, :] | (low[:, :, 1, :] << 4)).reshape((qs.shape[0], 128))


def _pack_q6k_qh(qs: torch.Tensor) -> torch.Tensor:
    assert qs.ndim == 3
    assert qs.shape[1:] == (16, 16)
    assert qs.dtype == torch.uint8
    assert torch.all(qs < 64)

    high = ((qs >> 4) & 0x03).reshape((qs.shape[0], 2, 4, 32))
    return (
        high[:, :, 0, :]
        | (high[:, :, 1, :] << 2)
        | (high[:, :, 2, :] << 4)
        | (high[:, :, 3, :] << 6)
    ).reshape((qs.shape[0], 64))


def make_patterned_q4k_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks

    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(8, dtype=torch.uint8).reshape((1, 8))
    lane = torch.arange(32, dtype=torch.uint8).reshape((1, 1, 32))

    raw = torch.empty((n_blocks, 144), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    md = torch.ones((n_blocks,), dtype=torch.float16)
    raw[:, 0:2] = dd.view(torch.uint8).reshape((n_blocks, 2))
    raw[:, 2:4] = md.view(torch.uint8).reshape((n_blocks, 2))

    # Unique per-subblock values make scale/min index mistakes show up as
    # simple 32-column bands in the diff plot.
    sc = (1 + subblock * 3 + (block_col.to(torch.uint8).reshape((n_blocks, 1)) % 3)).to(torch.uint8)
    mins = (2 + subblock * 5 + (block_row.to(torch.uint8).reshape((n_blocks, 1)) % 5)).to(torch.uint8)
    raw[:, 4:16] = _pack_q4k_scales_and_mins(sc, mins)

    # The q pattern distinguishes low/high nibble selection from scale/min
    # selection: adjacent 32-value subblocks are different, and lanes still vary.
    lane_half = lane // 16
    qs = ((subblock.reshape((1, 8, 1)) * 3 + lane + lane_half * 5) % 16).to(torch.uint8).expand(
        (n_blocks, 8, 32)
    ).clone()
    raw[:, 16:] = _pack_q4k_qs(qs)

    gguf_tensor = GGUFTensor(
        name="synthetic.patterned.q4_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q4_K,
        data=raw.numpy().tobytes(),
    )
    debug = {
        "dd": dd.reshape((rows, n_col_blocks)),
        "md": md.reshape((rows, n_col_blocks)),
        "sc": sc.reshape((rows, n_col_blocks, 8)),
        "mins": mins.reshape((rows, n_col_blocks, 8)),
        "qs": qs.reshape((rows, n_col_blocks, 8, 32)),
    }
    return gguf_tensor, debug


def make_patterned_q5k_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks

    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(8, dtype=torch.uint8).reshape((1, 8))
    lane = torch.arange(32, dtype=torch.uint8).reshape((1, 1, 32))
    lane_half = lane // 16

    raw = torch.empty((n_blocks, 176), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    md = torch.ones((n_blocks,), dtype=torch.float16)
    raw[:, 0:2] = dd.view(torch.uint8).reshape((n_blocks, 2))
    raw[:, 2:4] = md.view(torch.uint8).reshape((n_blocks, 2))

    sc = (3 + subblock * 4 + (block_col.to(torch.uint8).reshape((n_blocks, 1)) % 5)).to(torch.uint8)
    mins = (1 + subblock * 6 + (block_row.to(torch.uint8).reshape((n_blocks, 1)) % 7)).to(torch.uint8)
    raw[:, 4:16] = _pack_q4k_scales_and_mins(sc, mins)

    # Q5_K keeps the low four bits packed like Q4_K and stores bit 4 in a
    # separate 32-byte high-bit plane, one byte per lane.
    qs = ((subblock.reshape((1, 8, 1)) * 5 + lane + lane_half * 9) % 32).to(torch.uint8).expand(
        (n_blocks, 8, 32)
    ).clone()
    raw[:, 16:48] = _pack_q5k_qh(qs)
    raw[:, 48:] = _pack_q4k_qs(qs & 0x0F)

    gguf_tensor = GGUFTensor(
        name="synthetic.patterned.q5_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q5_K,
        data=raw.numpy().tobytes(),
    )
    debug = {
        "dd": dd.reshape((rows, n_col_blocks)),
        "md": md.reshape((rows, n_col_blocks)),
        "sc": sc.reshape((rows, n_col_blocks, 8)),
        "mins": mins.reshape((rows, n_col_blocks, 8)),
        "qs": qs.reshape((rows, n_col_blocks, 8, 32)),
    }
    return gguf_tensor, debug


def make_patterned_q6k_tensor(rows: int, cols: int) -> tuple[GGUFTensor, dict[str, torch.Tensor]]:
    assert cols % 256 == 0
    n_col_blocks = cols // 256
    n_blocks = rows * n_col_blocks

    block_row = torch.arange(n_blocks, dtype=torch.int64) // n_col_blocks
    block_col = torch.arange(n_blocks, dtype=torch.int64) % n_col_blocks
    subblock = torch.arange(16, dtype=torch.uint8).reshape((1, 16))
    lane = torch.arange(16, dtype=torch.uint8).reshape((1, 1, 16))
    lane_half = lane // 8

    raw = torch.empty((n_blocks, 210), dtype=torch.uint8)

    dd = torch.ones((n_blocks,), dtype=torch.float16)
    sc = (torch.arange(16, dtype=torch.int16).reshape((1, 16)) * 7 - 48).expand((n_blocks, 16)).clone()
    sc += (block_col.reshape((n_blocks, 1)) % 5).to(torch.int16)
    sc -= (block_row.reshape((n_blocks, 1)) % 3).to(torch.int16)
    sc = sc.clamp(-128, 127).to(torch.int8)

    qs = ((subblock.reshape((1, 16, 1)) * 5 + lane + lane_half * 13) % 64).to(torch.uint8).expand(
        (n_blocks, 16, 16)
    ).clone()

    raw[:, 0:128] = _pack_q6k_ql(qs)
    raw[:, 128:192] = _pack_q6k_qh(qs)
    raw[:, 192:208] = sc.view(torch.uint8)
    raw[:, 208:210] = dd.view(torch.uint8).reshape((n_blocks, 2))

    gguf_tensor = GGUFTensor(
        name="synthetic.patterned.q6_k",
        shape=[cols, rows],
        dtype=GGUFType.GGML_TYPE_Q6_K,
        data=raw.numpy().tobytes(),
    )
    debug = {
        "dd": dd.reshape((rows, n_col_blocks)),
        "sc": sc.reshape((rows, n_col_blocks, 16)),
        "qs": qs.reshape((rows, n_col_blocks, 16, 16)),
    }
    return gguf_tensor, debug


def plot_heatmap(
    x: torch.Tensor,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    annotate: bool = False,
    fmt: str = ".2f",
    colorbar: bool = True,
    cmap: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")

    data = x.detach().float().cpu().numpy()
    rows, cols = data.shape

    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticklabels is not None:
        if len(xticklabels) != cols:
            raise ValueError(f"xticklabels has length {len(xticklabels)}, expected {cols}")
        ax.set_xticks(range(cols))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    if yticklabels is not None:
        if len(yticklabels) != rows:
            raise ValueError(f"yticklabels has length {len(yticklabels)}, expected {rows}")
        ax.set_yticks(range(rows))
        ax.set_yticklabels(yticklabels)

    if annotate:
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")

    if colorbar:
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return fig, ax


QUANT_TESTS = {
    "q4_k": {
        "dtype": tensor.q4_k.name,
        "dequant": dequant_q4k_to_fp32,
        "report_group_cols": 32,
        "inputs": {
            "patterned": make_patterned_q4k_tensor,
            "random": make_random_q4k_test_tensor,
        },
    },
    "q5_k": {
        "dtype": tensor.q5_k.name,
        "dequant": dequant_q5k_to_fp32,
        "report_group_cols": 32,
        "inputs": {
            "patterned": make_patterned_q5k_tensor,
            "random": make_random_q5k_test_tensor,
        },
    },
    "q6_k": {
        "dtype": tensor.q6_k.name,
        "dequant": dequant_q6k_to_fp32,
        "report_group_cols": 16,
        "inputs": {
            "patterned": make_patterned_q6k_tensor,
            "random": make_random_q6k_test_tensor,
        },
    },
}


SHAPE_TEST_CASES = [
    # Small, surgical cases. BLOCK_COLS=16 is currently unsupported.
    # (1, 256, 1, 16),
    (1, 256, 1, 32),
    (1, 256, 1, 64),
    (1, 256, 1, 128),
    # Matmul-like K tile sizes from the autotune configs.
    # (128, 256, 32, 16),
    (128, 256, 32, 32),
    # (128, 256, 64, 16),
    (128, 256, 64, 64),
    # (128, 256, 128, 16),
    (128, 256, 128, 128),
    # Multiple quantized superblocks along N catches superblock offset mistakes.
    # (128, 512, 64, 16),
    (128, 512, 64, 64),
    (128, 512, 128, 128),
]


TEST_CASES = [
    (quant_name, input_name, *shape_case)
    for quant_name, quant in QUANT_TESTS.items()
    for input_name in quant["inputs"]
    for shape_case in SHAPE_TEST_CASES
]


def run_case(quant_name: str, input_name: str, rows: int, cols: int, block_rows: int, block_cols: int):
    assert rows % block_rows == 0
    assert cols % block_cols == 0

    quant = QUANT_TESTS[quant_name]
    gguf_tensor, debug = quant["inputs"][input_name](rows, cols)
    q_cpu = torch.frombuffer(bytearray(gguf_tensor.data), dtype=torch.uint8).reshape(rows, -1)
    q = q_cpu.to("cuda")
    out = torch.empty((rows, cols), dtype=torch.float32, device="cuda")
    ref = quant["dequant"](gguf_tensor).reshape(rows, cols).to("cuda")

    grid = lambda META: (
        triton.cdiv(out.shape[1], META["BLOCK_COLS"]),
        triton.cdiv(out.shape[0], META["BLOCK_ROWS"]),
    )

    launch[_dequant_loader_test_kernel, grid](
        out=out,
        q=q,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        CONCEPTUAL_DTYPE=quant["dtype"],
    )
    cuda_sync()
    torch.cuda.synchronize()

    diff = (out - ref).abs()
    diff_vs_fp16_ref = (out - ref.to(torch.float16).to(torch.float32)).abs()
    report_group_cols = quant["report_group_cols"]
    subblock_diff = diff.reshape((rows, cols // report_group_cols, report_group_cols)).max(-1).values
    pooled_rows = max(1, rows // block_rows)
    pooled_cols = max(1, cols // block_cols)
    tile_diff = diff.reshape((pooled_rows, block_rows, pooled_cols, block_cols)).transpose(1, 2).reshape(
        (pooled_rows, pooled_cols, -1)
    ).max(-1).values

    return {
        "quant_name": quant_name,
        "input_name": input_name,
        "rows": rows,
        "cols": cols,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "report_group_cols": report_group_cols,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "max_diff_vs_fp16_ref": diff_vs_fp16_ref.max().item(),
        "mean_diff_vs_fp16_ref": diff_vs_fp16_ref.mean().item(),
        "real_abs_max": out.abs().max().item(),
        "expected_abs_max": ref.abs().max().item(),
        "real_std": out.float().std(unbiased=False).item(),
        "expected_std": ref.float().std(unbiased=False).item(),
        "custom_min": out.min().item(),
        "custom_max": out.max().item(),
        "ref_min": ref.min().item(),
        "ref_max": ref.max().item(),
        "diff": diff,
        "subblock_diff": subblock_diff,
        "tile_diff": tile_diff,
        "debug": debug,
    }


results = []
for case in TEST_CASES:
    results.append(run_case(*case))

failed = [result for result in results if result["max_diff"] != 0.0]
worst = max(results, key=lambda result: result["max_diff"])
print()
print("dequant test report")
print(f"total cases: {len(results)}")
print(f"zero-diff cases: {len(results) - len(failed)}")
print(f"nonzero-diff cases: {len(failed)}")

if failed:
    print()
    print("nonzero diff cases")
    for result in failed:
        print(
            f"- quant={result['quant_name']} input={result['input_name']} rows={result['rows']} cols={result['cols']} "
            f"block_rows={result['block_rows']} block_cols={result['block_cols']} "
            f"max_diff={result['max_diff']:.8g} mean_diff={result['mean_diff']:.8g} "
            f"max_diff_vs_fp16_ref={result['max_diff_vs_fp16_ref']:.8g} "
            f"expected_abs_max={result['expected_abs_max']:.8g} real_abs_max={result['real_abs_max']:.8g} "
            f"expected_std={result['expected_std']:.8g} real_std={result['real_std']:.8g}"
        )
        print(f"  max abs diff by logical {result['report_group_cols']}-value subblock")
        print(result["subblock_diff"])
        print("  max abs diff by launched tile")
        print(result["tile_diff"])

    print()
    print(
        f"worst case quant={worst['quant_name']} input={worst['input_name']} rows={worst['rows']} cols={worst['cols']} "
        f"block_rows={worst['block_rows']} block_cols={worst['block_cols']}"
    )
    if worst["debug"]:
        print("pattern sc row0/block0", worst["debug"]["sc"][0, 0])
        if "mins" in worst["debug"]:
            print("pattern mins row0/block0", worst["debug"]["mins"][0, 0])
        print("pattern qs row0/block0/subblocks")
        print(worst["debug"]["qs"][0, 0])

    plot_heatmap(
        worst["diff"],
        title=(
            f"{worst['quant_name']} {worst['input_name']} dequant abs diff worst case "
            f"{worst['rows']}x{worst['cols']} tile {worst['block_rows']}x{worst['block_cols']}"
        ),
        xlabel="col",
        ylabel="row",
        cmap="viridis",
    )

    plot_heatmap(
        worst["subblock_diff"],
        title=(
            f"{worst['quant_name']} {worst['input_name']} dequant max abs diff "
            f"by {worst['report_group_cols']}-value subblock"
        ),
        xlabel="logical subblock",
        ylabel="row",
        xticklabels=[str(i) for i in range(worst["cols"] // worst["report_group_cols"])],
        cmap="viridis",
    )

    plot_heatmap(
        worst["tile_diff"],
        title="dequant max-pooled abs diff by launched tile",
        xlabel=f"col blocks x{worst['block_cols']}",
        ylabel=f"row blocks x{worst['block_rows']}",
        cmap="viridis",
    )
else:
    print("all tested cases matched exactly; no heatmaps generated")

g4b.device.teardown()
