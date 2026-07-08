#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <format>
#include <thread>
#include <utility>
#include <mutex>
#include <atomic>
#include <numeric>
#include <limits>
#include <condition_variable>
#include <latch>
#include <queue>
#include <unordered_map>
#include <variant>
#include <exception>
#include <cstdint>
#include <cstddef>
#include <cassert>

template<typename T>
class ThreadSafeQueue {
	std::mutex m_mutex;
	std::queue<T> m_queue;
	std::condition_variable m_condition;
	std::atomic_bool m_interrupted;

public:
	struct InterruptedError : std::exception {
	};

	ThreadSafeQueue() = default;

	T get() {
		std::unique_lock lk(m_mutex);
		m_condition.wait(lk, [this] { return !m_queue.empty() || m_interrupted.load(); });
		if (m_interrupted.load())
			throw InterruptedError{};
		const T out = m_queue.front();
		m_queue.pop();
		return out;
	}

	void put_all(std::span<T> values) {
		std::lock_guard lk(m_mutex);
		for (T &t: values)
			m_queue.push(t);
		m_condition.notify_all();
	}

	void interrupt() {
		m_interrupted.store(true);
	}
};

using utf32_t = uint32_t;
using token_t = uint32_t;
using token_pair_t = uint64_t;

static inline token_pair_t make_token_pair(const token_t t1, const token_t t2) {
	return (static_cast<token_pair_t>(t2) << (8 * sizeof(token_t))) | static_cast<token_pair_t>(t1);
}

class Tokenizer {
	struct Job {
		uint64_t submission_id{};
		uint64_t job_id{};
		std::u32string_view seq;
	};

	struct Solution {
		uint64_t job_id{};
		std::vector<token_t> seq;
	};

	struct Submission {
		uint64_t id;
		std::latch jobs_left;
		std::vector<Solution> solutions{};
		std::mutex solutions_mutex{};

		Submission(uint64_t id, uint64_t n_jobs);
	};

	std::unordered_map<uint64_t, Submission> m_submissions{};
	std::mutex m_submissions_mutex{};
	std::atomic_uint64_t m_next_submission_id{};

	ThreadSafeQueue<Job> m_jobs{};
	std::vector<std::thread> m_workers{};
	std::atomic_uint64_t m_num_workers_exited{};

	// Constant-after-load tokenizer state
	std::unordered_map<std::u32string_view, token_t> m_str_to_tok{};
	std::unordered_map<std::u32string_view, token_t> m_special_str_to_tok{};
	uint64_t m_max_special_token_str_len;
	std::array<token_t, 256> m_byte_to_tok{};
	std::unordered_map<token_pair_t, std::tuple<uint32_t, token_t> > m_merges{};
	token_t m_eos_id;
	std::vector<std::u32string> m_string_pool{};

	std::vector<token_t> merge(std::u32string_view seq);

	void worker();

	static void worker_wrapper(Tokenizer *self);

public:
	Tokenizer(std::span<std::span<utf32_t> > tokens, std::span<token_t> token_types,
	          std::span<token_t> merges, token_t eos_id);

	~Tokenizer();

	std::tuple<token_t *, uint64_t> tokenize(std::u32string_view seq, bool allow_special);
};

extern "C" void *create_tokenizer(utf32_t **tokens, const uint64_t *token_lengths, token_t *token_types,
                                  const uint64_t token_count, token_t *merges, const uint64_t merge_count,
                                  const token_t eos_id) {
	std::vector<std::span<utf32_t> > tokens_;
	tokens_.reserve(token_count);
	for (size_t i = 0; i < token_count; i++)
		tokens_.emplace_back(tokens[i], token_lengths[i]);
	return new Tokenizer{tokens_, std::span(token_types, token_count), std::span(merges, merge_count), eos_id};
}

extern "C" void destroy_tokenizer(const void *tokenizer) {
	delete static_cast<const Tokenizer *>(tokenizer);
}

extern "C" void tokenize(void *tokenizer, const utf32_t *sequence, const uint64_t length, token_t **out,
                         uint64_t *out_length, const int32_t allow_special) {
	auto *tokenizer_ = static_cast<Tokenizer *>(tokenizer);
	const std::u32string_view seq{reinterpret_cast<const char32_t *>(sequence), length};
	auto [out_p, out_len] = tokenizer_->tokenize(seq, allow_special);
	*out = out_p;
	*out_length = out_len;
}

extern "C" void destroy_tokens(const token_t *tokens) {
	delete[] tokens;
}

std::tuple<token_t *, uint64_t> Tokenizer::tokenize(std::u32string_view seq, const bool allow_special) {
	const uint64_t submission_id = m_next_submission_id.fetch_add(1, std::memory_order::relaxed);

	// Split into chunks at runs of newlines and if allow_special, also at special tokens
	std::vector<std::variant<token_t, Job> > tokens_or_jobs;
	{
		uint64_t next_job_id = 0;
		std::u32string_view seq_after_prev_split = seq;
		const auto append_normal_job_since_prev_split = [
					&tokens_or_jobs, &next_job_id, &seq, &seq_after_prev_split, submission_id
				] {
			tokens_or_jobs.emplace_back(Job{
				submission_id,
				next_job_id++,
				seq_after_prev_split.substr(0, seq_after_prev_split.length() - seq.length())
			});
		};
		while (!seq.empty()) {
			if (seq.starts_with('\n')) {
				append_normal_job_since_prev_split();
				// Consume run of newlines
				const auto seq_orig = seq;
				while (seq.starts_with('\n'))
					seq = seq.substr(1);
				decltype(m_str_to_tok)::iterator it;
				if (const auto newlines = seq_orig.substr(0, seq_orig.length() - seq.length());
					(it = m_str_to_tok.find(newlines)) != m_str_to_tok.end()) {
					tokens_or_jobs.emplace_back(it->second);
				} else {
					tokens_or_jobs.emplace_back(Job{submission_id, next_job_id++, newlines});
				}
				seq_after_prev_split = seq;
				continue;
			}
			if (allow_special) {
				// Try consuming a special token
				std::u32string_view head = seq.substr(0, m_max_special_token_str_len);
				while (!head.empty()) {
					if (decltype(m_special_str_to_tok)::iterator it;
						(it = m_special_str_to_tok.find(head)) != m_special_str_to_tok.end()) {
						append_normal_job_since_prev_split();
						tokens_or_jobs.emplace_back(it->second);
						seq = seq.substr(head.length());
						seq_after_prev_split = seq;
						continue;
					}
					head = head.substr(0, head.length() - 1);
				}
			}
			// Normal char
			seq = seq.substr(1);
		}
	}
	std::vector<Job> jobs{};
	for (const auto &x: tokens_or_jobs) {
		if (std::holds_alternative<Job>(x)) {
			jobs.emplace_back(std::get<Job>(x));
		}
	}

	// Create submission
	{
		std::lock_guard lk(m_submissions_mutex);
		m_submissions.erase(submission_id);
		m_submissions.try_emplace(submission_id, submission_id, jobs.size());
	}
	auto &submission = m_submissions.at(submission_id);

	// Enqueue jobs
	m_jobs.put_all(jobs);

	// Await completion by workers
	submission.jobs_left.wait();

	// Collect results and assemble the final output sequence
	token_t *out;
	size_t out_i;
	{
		std::lock_guard lk(submission.solutions_mutex);
		std::sort( // Sort to prepare for merge with tokens_or_jobs
			submission.solutions.begin(), submission.solutions.end(),
			[](const Solution &a, const Solution &b) { return a.job_id < b.job_id; }
		);

		// Allocate output buffer
		size_t n_output_tokens =
				std::count_if(
					tokens_or_jobs.begin(), tokens_or_jobs.end(),
					[](const auto &x) { return std::holds_alternative<token_t>(x); }
				) + std::accumulate(
					submission.solutions.begin(), submission.solutions.end(), 0,
					[](const auto accum, const auto &sol) {
						return accum + sol.seq.size();
					}
				);
		out = new token_t[n_output_tokens];
		out_i = 0;
		const auto append_to_out = [&out, &out_i, &n_output_tokens](const std::span<token_t> tokens) {
			if (out_i + tokens.size() >= n_output_tokens) {
				// Reallocate
				size_t new_n_output_tokens = n_output_tokens;
				while (out_i + tokens.size() >= new_n_output_tokens)
					new_n_output_tokens *= 2;
				auto *new_out = new token_t[new_n_output_tokens];
				std::copy_n(out, out_i, new_out);
				delete[] out;
				out = new_out;
				n_output_tokens = new_n_output_tokens;
			}
			std::ranges::copy(tokens, &out[out_i]);
			out_i += tokens.size();
		};

		// Merge in one pass
		size_t solution_i = 0, toj_i = 0;
		while (toj_i < tokens_or_jobs.size()) {
			assert(solution_i <= submission.solutions.size());

			if (std::holds_alternative<Job>(tokens_or_jobs.at(toj_i))) {
				const std::span new_tokens = submission.solutions.at(solution_i).seq;
				append_to_out(new_tokens);
				solution_i++, toj_i++;
			} else {
				std::array new_tokens{std::get<token_t>(tokens_or_jobs.at(toj_i))};
				append_to_out(new_tokens);
				toj_i++;
			}
		}
		assert(solution_i == submission.solutions.size());
	}

	{
		std::lock_guard lk(m_submissions_mutex);
		m_submissions.erase(submission_id);
	}

	return {out, out_i};
}

static size_t unicode_codepoint_to_utf8(uint8_t buffer[4], const uint32_t code) {
	if (code <= 0x7F) {
		buffer[0] = code;
		return 1;
	}
	if (code <= 0x7FF) {
		buffer[0] = 0xC0 | (code >> 6); // 110xxxxx
		buffer[1] = 0x80 | (code & 0x3F); // 10xxxxxx
		return 2;
	}
	if (code <= 0xFFFF) {
		buffer[0] = 0xE0 | (code >> 12); // 1110xxxx
		buffer[1] = 0x80 | ((code >> 6) & 0x3F); // 10xxxxxx
		buffer[2] = 0x80 | (code & 0x3F); // 10xxxxxx
		return 3;
	}
	if (code <= 0x10FFFF) {
		buffer[0] = 0xF0 | (code >> 18); // 11110xxx
		buffer[1] = 0x80 | ((code >> 12) & 0x3F); // 10xxxxxx
		buffer[2] = 0x80 | ((code >> 6) & 0x3F); // 10xxxxxx
		buffer[3] = 0x80 | (code & 0x3F); // 10xxxxxx
		return 4;
	}
	return 0;
}

std::vector<token_t> Tokenizer::merge(const std::u32string_view seq) {
	assert(seq.length() <= std::numeric_limits<uint32_t>::max());
	constexpr uint32_t token_is_unknown_unicode_bit = 1u << 31;

	struct Piece {
		token_t token;
		uint32_t prev;
		uint32_t next;
	};

	struct Merge {
		uint32_t rank;
		uint32_t i;
		uint32_t a;
		uint32_t b;
	};

	auto merge_cmp = [](const Merge &l, const Merge &r) { return l.rank < r.rank; };

	auto try_add_merge = [](
		std::priority_queue<Merge, std::vector<Merge>, decltype(merge_cmp)> &merges, const uint32_t ai, const Piece *a,
		const Piece *b, const std::unordered_map<token_pair_t, std::tuple<uint32_t, token_t> > &merge_map
	) {
		if (!a || !b) return;
		const auto m = merge_map.find(make_token_pair(a->token, b->token));
		if (m == merge_map.end()) return;
		const uint32_t rank = std::get<0>(m->second);
		merges.push(Merge{rank, ai, a->token, b->token});
	};

	std::vector<Piece> pieces;
	std::priority_queue<Merge, std::vector<Merge>, decltype(merge_cmp)> merges{merge_cmp};
	pieces.reserve(seq.length());

	token_t prev_token = -1;
	for (uint32_t i = 0; i < seq.length(); i++) {
		token_t token;
		if (const auto it = m_str_to_tok.find(seq.substr(i, 1));
			it != m_str_to_tok.end()) {
			token = it->second;
		} else {
			// Code point is not a valid token. Will need to decompose into bytes later. However, I must not do it now
			//  since that may cause merging with other tokens, which is incorrect tokenization (even if subtle).
			token = static_cast<uint32_t>(seq[i]) | token_is_unknown_unicode_bit;
		}
		uint32_t prev = i - 1; // overflows to UINT32_MAX naturally
		uint32_t next = i + 1 < seq.length() ? i + 1 : -1;
		pieces.emplace_back(token, prev, next);

		if (const auto it = m_merges.find(make_token_pair(prev_token, token));
			it != m_merges.end()) {
			const uint32_t rank = std::get<0>(it->second);
			merges.push(Merge{rank, i - 1, prev_token, token});
		}
		prev_token = token;
	}

	while (!merges.empty()) {
		const Merge merge = merges.top();
		merges.pop();

		// Load and check if stale
		Piece &a = pieces.at(merge.i);
		if (a.token != merge.a) continue;
		Piece &b = pieces.at(a.next);
		if (b.token != merge.b) continue;

		// Check if mergeable
		const auto m = m_merges.find(make_token_pair(a.token, b.token));
		if (m == m_merges.end()) continue;

		// Merge
		a.token = std::get<1>(m->second);
		b.token = -1;

		// Unlink b
		a.next = b.next;
		Piece *c = b.next != -1 ? &pieces.at(b.next) : nullptr;
		if (c) c->prev = b.prev;
		b.next = -1;
		b.prev = -1;

		// Try to add merges if pieces have neighbors
		if (const Piece *aa = a.prev != -1 ? &pieces.at(a.prev) : nullptr)
			try_add_merge(merges, a.prev, aa, &a, m_merges);
		if (c)
			try_add_merge(merges, merge.i, &a, c, m_merges);
	}

	std::vector<token_t> out;
	out.reserve(pieces.size() + pieces.size() / 8); // alloc with safety margin for splitting some tokens into bytes
	for (const Piece &p: pieces) {
		if (p.token == -1) continue;
		if (p.token & token_is_unknown_unicode_bit) {
			// This is not actually a token, it's just a Unicode codepoint without a corresponding token.
			//  Must be decomposed into tokens representing its individual bytes depending on how many it has.
			const token_t codepoint = p.token & ~token_is_unknown_unicode_bit;
			uint8_t buffer[4];
			const size_t n_bytes = unicode_codepoint_to_utf8(buffer, codepoint);
			for (size_t i = 0; i < n_bytes; i++) {
				out.emplace_back(m_byte_to_tok[buffer[i]]);
			}
		} else {
			out.emplace_back(p.token);
		}
	}

	out.shrink_to_fit();
	return out;
}

void Tokenizer::worker() {
	const auto submit = [this](const Job &job, std::vector<token_t> seq) {
		std::lock_guard lk(m_submissions_mutex);
		Submission &submission = m_submissions.at(job.submission_id);
		submission.solutions.emplace_back(job.job_id, std::move(seq));
		submission.jobs_left.count_down();
	};

	while (true) {
		// Fetch job
		Job job;
		try {
			job = m_jobs.get();
		} catch (ThreadSafeQueue<Job>::InterruptedError &) {
			// Worker should exit
			break;
		}

		// Process
		assert(!job.seq.empty());
		if (decltype(m_str_to_tok)::iterator it;
			job.seq.at(0) == '\n' && (it = m_str_to_tok.find(job.seq)) != m_str_to_tok.end()) {
			// Newline sequences may not have bpe merge entries but still a token entry
			submit(job, {it->second});
		} else {
			submit(job, merge(job.seq));
		}
	}
	m_num_workers_exited.fetch_add(1);
	m_num_workers_exited.notify_all();
}

Tokenizer::Tokenizer(const std::span<std::span<utf32_t> > tokens, const std::span<token_t> token_types,
                     const std::span<token_t> merges, const token_t eos_id)
	: m_max_special_token_str_len(0), m_eos_id(eos_id) {
	for (size_t i = 0; i < tokens.size(); i++) {
		const std::u32string k{tokens[i].begin(), tokens[i].end()};
		m_string_pool.emplace_back(k);
		m_str_to_tok.emplace(m_string_pool.at(m_string_pool.size() - 1), i);
		if (token_types[i] != 1) {
			m_special_str_to_tok.emplace(m_string_pool.at(m_string_pool.size() - 1), i);
			m_max_special_token_str_len = std::max(m_max_special_token_str_len, k.length());
		}
	}

	for (size_t i = 0; i < 256; i++) {
		std::string hex = std::format("<0x{:02X}>", i);
		m_byte_to_tok[i] = m_str_to_tok.at(std::u32string(hex.begin(), hex.end()));
	}

	m_merges.reserve(merges.size());
	for (size_t i = 0; i < merges.size(); i += 3)
		m_merges.emplace(make_token_pair(merges[i], merges[i + 1]), std::tuple{i / 3, merges[i + 2]});

	const size_t n_workers = std::max(2 * std::thread::hardware_concurrency() / 3, 1u);
	m_workers.reserve(n_workers);
	for (size_t i = 0; i < n_workers; i++)
		m_workers.emplace_back(worker_wrapper, this);

	assert(m_str_to_tok.size() <= std::numeric_limits<uint32_t>::max() / 4); // upper bits used by tokenizer
}

Tokenizer::~Tokenizer() {
	// Terminate the workers gracefully
	m_jobs.interrupt();
	for (uint64_t num_workers_exited = 0; num_workers_exited != m_workers.size();
	     num_workers_exited = m_num_workers_exited.load()) {
		m_num_workers_exited.wait(num_workers_exited);
	}
}

Tokenizer::Submission::Submission(const uint64_t id, const uint64_t n_jobs)
	: id(id), jobs_left(std::latch(static_cast<std::ptrdiff_t>(n_jobs))) {
	solutions.reserve(n_jobs);
}

void Tokenizer::worker_wrapper(Tokenizer *self) {
	self->worker();
}
