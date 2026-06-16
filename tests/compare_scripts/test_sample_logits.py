import torch
import time
import g4b.device
from g4b.utils import to_int_exact
from g4b.kernels.utils import launch
from g4b.kernels.sample_logits import sample_logits, get_recommended_num_v_splits, _bitonic_reduce_jfn

# TODO this kernel is still a little ~1.1x slower than pytorch for bf16

g4b.device.init(0)

import triton.runtime.driver

triton.runtime.driver.active.utils.set_printf_fifo_size(1024 * 1024 * 256)

cuda_sync = lambda: g4b.device.stream.sync()

# B, T, V = 300, 17, 256
# B, T, V = 1, 1, 512
B, T, V = 1, 1, 2 ** 18

# embed_dtype = torch.float32
# embed_dtype = torch.float16
embed_dtype = torch.bfloat16
# embed_dtype = torch.int8

temperature = 0.9
top_k = 64
top_p = 0.8
logit_softcap = 10.

NUM_V_SPLITS = get_recommended_num_v_splits(V)
print("num v splits:", NUM_V_SPLITS)

logits = torch.randn(B, T, V, dtype=embed_dtype, device="cuda")
sampled_tokens = torch.randint(0, V, (B, T), dtype=torch.int32, device="cuda")
seed = torch.tensor([0, 0], dtype=torch.int32, device="cuda")

_scratchpad_shape = (B, T, NUM_V_SPLITS, top_k)
tmp_top_k_logits_scratchpad = torch.empty(_scratchpad_shape, dtype=torch.float32, device="cuda")
tmp_top_k_idx_scratchpad = torch.empty(_scratchpad_shape, dtype=torch.int32, device="cuda")

_torch_seed, _torch_base_offs = 0, 0
_torch_rands = torch.randn(B, T, dtype=embed_dtype, device="cuda")
logits_orig = logits.clone()
sampled_tokens_orig = sampled_tokens.clone()
seed_orig = seed.clone()


def restore_inputs():
    global logits, sampled_tokens, seed, _torch_seed, _torch_base_offs
    logits = logits_orig.clone()
    sampled_tokens = sampled_tokens_orig.clone()
    seed = seed_orig.clone()
    _torch_seed = 0
    _torch_base_offs = 0
    torch.cuda.synchronize()


def do_forward():
    sample_logits(
        logits,
        sampled_tokens,
        seed,
        tmp_top_k_logits_scratchpad,
        tmp_top_k_idx_scratchpad,
        temperature,
        top_k,
        top_p,
        NUM_V_SPLITS=NUM_V_SPLITS,
    )


import triton
import triton.language as tl


@triton.jit
def _gen_rands_kernel(
    # fmt: off
    out_rands_ptr, seed, offs_base,
    out_rands_shape0: tl.constexpr, out_rands_shape1: tl.constexpr,
    out_rands_stride0: tl.constexpr, out_rands_stride1: tl.constexpr,
    BLOCKSIZE0: tl.constexpr = 128, BLOCKSIZE1: tl.constexpr = 128,
    # fmt: on
):
    B: tl.constexpr = out_rands_shape0
    T: tl.constexpr = out_rands_shape1

    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    off_b = pid_b * BLOCKSIZE0 + tl.arange(0, BLOCKSIZE0)[:, None]
    off_t = pid_t * BLOCKSIZE1 + tl.arange(0, BLOCKSIZE1)[None, :]
    offs = off_b * T + off_t  # sampling grid with fake-contiguous striding (-> samples independent of mem layout)
    rands = tl.rand(seed, offs_base + offs)

    out_rands_offs = off_b * out_rands_stride0 + off_t * out_rands_stride1
    tl.store(out_rands_ptr + out_rands_offs, rands, mask=(off_b < B) & (off_t < T))


def _gen_rands(out_rands, seed: int, base_offs: int):
    grid_fn = lambda META: (
        triton.cdiv(out_rands.shape[1], META["BLOCKSIZE1"]),
        triton.cdiv(out_rands.shape[0], META["BLOCKSIZE0"]),
    )
    return launch[_gen_rands_kernel, grid_fn](
        out_rands=out_rands,
        seed=to_int_exact(seed),
        offs_base=to_int_exact(base_offs),
    )


def _top_k_top_p_filtering(logits, rands, top_k=0, top_p=0.0):
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    assert rands.dim() == 1
    top_k_logits, top_k_idx = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)

    probs = torch.softmax(top_k_logits / temperature, dim=-1)
    p_cumsum = probs.cumsum(dim=-1)
    probs = torch.where(p_cumsum - probs <= top_p, probs, 0.0)
    probs /= probs.sum(dim=-1, keepdim=True)
    p_cumsum = probs.cumsum(dim=-1)

    sampled_top_k_idx = (rands[:, None] <= p_cumsum).to(torch.int32).argmax(dim=-1)
    return top_k_idx.gather(1, sampled_top_k_idx[:, None]).to(torch.int32)


def do_forward_torch():
    global sampled_tokens, _torch_seed, _torch_base_offs
    _gen_rands(_torch_rands, _torch_seed, _torch_base_offs)
    _torch_seed += 7 * T
    _torch_base_offs += 11 * T

    logits_ = logits.reshape((B * T, V))
    if logit_softcap is not None:
        logits_ = logit_softcap * (logits_ / logit_softcap).tanh()
    sampled_tokens = _top_k_top_p_filtering(
        logits_,
        _torch_rands.reshape((B * T,)),
        top_k,
        top_p,
    ).reshape((B, T))
    # top_k_logits, top_k_idx = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
    # global tmp_top_k_logits_scratchpad
    # tmp_top_k_logits_scratchpad = top_k_logits.reshape((B, T, 1, top_k))


def capture_out():
    #return tmp_top_k_logits_scratchpad
    return sampled_tokens


# @triton.jit
# def _bitonic_test_kernel():
#     N: tl.constexpr = 128
#     a = tl.arange(10, 10 + N).to(tl.float32)[None, None, :]
#     b = tl.arange(20, 20 + N).to(tl.float32)[None, None, :]
#     a_idx = tl.arange(0, N)[None, None, :]
#     b_idx = tl.arange(200, 200 + N)[None, None, :]
#     ac, ac_idx = _bitonic_reduce_jfn(a, a_idx, b, b_idx)
#     print("ac", ac)
#     print("ac", ac_idx)
#
#
# _bitonic_test_kernel[(1,)]()
# exit()


N_reps = 1000

# warmup
torch.set_float32_matmul_precision("medium")
torch.cuda.synchronize()
restore_inputs()
for _ in range(N_reps):
    do_forward()
cuda_sync()
restore_inputs()
for _ in range(N_reps):
    do_forward_torch()
torch.cuda.synchronize()

# measure custom
restore_inputs()
start = time.time()
for _ in range(N_reps):
    do_forward()
out_custom = capture_out()
cuda_sync()
end = time.time()
print("custom", end - start)

# measure torch
restore_inputs()
start = time.time()
for _ in range(N_reps):
    do_forward_torch()
out_torch = capture_out()
torch.cuda.synchronize()
end = time.time()
print("torch", end - start)

print("abs diff", (out_custom - out_torch).abs().max())
print("custom abs max", out_custom.abs().max())
print("torch abs max", out_torch.abs().max())
print("custom min/max", out_custom.min(), out_custom.max())


from pathlib import Path
from typing import Optional, Sequence

import torch
import matplotlib.pyplot as plt


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
    """
    Plot a 2D torch tensor as a heatmap.

    Args:
        x: 2D torch tensor.
        title: Optional plot title.
        xlabel/ylabel: Optional axis labels.
        xticklabels/yticklabels: Optional labels for columns/rows.
        annotate: If True, write values into cells.
        fmt: Format string for annotations.
        colorbar: Whether to show a colorbar.
        cmap: Optional matplotlib colormap name, e.g. "viridis", "magma".
        save_path: Optional path to save the figure.
        show: Whether to call plt.show().

    Returns:
        fig, ax
    """
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


plot_heatmap(
    ((out_custom - out_torch).abs() + 1).reshape((B, -1)),
    title="Sampled token diff",
    xlabel="t",
    ylabel="b",
    cmap="viridis",
)

g4b.device.teardown()
