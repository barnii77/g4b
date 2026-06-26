#ifdef _MSC_VER
#define _export __cdecl
#else
#define _export __attribute__((cdecl))
#endif

struct Tokenizer {
	...  // TODO data
};

_export Tokenizer *create(...) {
	return new Tokenizer(...);
}

_export void destroy(Tokenizer *tokenizer) {
	delete tokenizer;
}

// TODO impl
