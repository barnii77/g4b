#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <format>
#include <thread>
#include <utility>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <cstdint>

#ifdef _MSC_VER
#define _export extern "C" __cdecl
#else
#define _export extern "C" __attribute__((cdecl))
#endif

template<typename T>
class ThreadSafeQueue {
	std::mutex m_mutex;
	std::queue<T> m_queue;
	std::condition_variable m_condition;

public:
	ThreadSafeQueue() = default;

	T get() {
		std::unique_lock lk(m_mutex);
		m_condition.wait(lk, [this] { return !m_queue.empty(); });
		const T out = m_queue.front();
		m_queue.pop();
		return out;
	}

	void putAll(std::span<T> values) {
		std::lock_guard lk(m_mutex);
		for (T &t: values)
			m_queue.push(t);
		m_condition.notify_all();
	}
};

using utf32_t = uint32_t;
using token_t = uint32_t;
using token_pair_t = uint64_t;

static inline token_pair_t make_token_pair(const token_t t1, const token_t t2) {
	return (static_cast<token_pair_t>(t2) << (8 * sizeof(token_t))) | static_cast<token_pair_t>(t1);
}

class Tokenizer {
	ThreadSafeQueue<std::u32string> m_jobs{};
	std::vector<std::thread> m_workers{};
	std::unordered_map<std::u32string, token_t> m_normal_str_to_tok{};
	std::unordered_map<std::u32string, token_t> m_special_str_to_tok{};
	std::array<token_t, 256> m_byte_to_tok{};
	std::unordered_map<token_pair_t, token_t> m_merges{};
	token_t m_eos_id;

public:
	Tokenizer(std::span<std::span<utf32_t> > tokens, std::span<token_t> token_types,
	          std::span<token_t> merges, token_t eos_id);
};

_export Tokenizer *create(utf32_t **tokens, const uint64_t *token_lengths, token_t *token_types,
                          const uint64_t token_count, token_t *merges, const uint64_t merge_count,
                          const token_t eos_id) {
	std::vector<std::span<utf32_t> > tokens_;
	tokens_.reserve(token_count);
	for (size_t i = 0; i < token_count; i++)
		tokens_.emplace_back(tokens[i], token_lengths[i]);
	return new Tokenizer{tokens_, std::span(token_types, token_count), std::span(merges, merge_count), eos_id};
}

_export void destroy(const Tokenizer *tokenizer) {
	delete tokenizer;
}

// TODO impl

Tokenizer::Tokenizer(const std::span<std::span<utf32_t> > tokens, const std::span<token_t> token_types,
                     const std::span<token_t> merges, const token_t eos_id) : m_eos_id(eos_id) {
	for (size_t i = 0; i < tokens.size(); i++) {
		const std::u32string k{tokens[i].begin(), tokens[i].end()};
		if (token_types[i] == 1)
			m_normal_str_to_tok.emplace(k, i);
		else
			m_special_str_to_tok.emplace(k, i);
	}

	for (size_t i = 0; i < 256; i++) {
		std::string hex = std::format("<0x{:02X}>", i);
		m_byte_to_tok[i] = m_normal_str_to_tok.at(std::u32string(hex.begin(), hex.end()));
	}

	m_merges.reserve(merges.size());
	for (size_t i = 0; i < merges.size(); i += 3)
		m_merges.emplace(make_token_pair(merges[i], merges[i + 1]), merges[i + 2]);
}
