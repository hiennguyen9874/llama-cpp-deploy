# CLI Configuration Options

[Guide][https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#usage]

## ctx-size and parralell

The --ctx-size argument actually specifies the total size of the KV cache (legacy name, --kv-size would be better). This corresponds to the total amount of tokens that can be stored across all independent sequences.

For example, if we specify --ctx-size 8192 this means that we can process:

2 sequences, each of max length of 4096 tokens
4 sequences, each of max length of 2048 tokens
8 sequences, each of max length of 1024 tokens
...
32 sequences, each of max length of 256 tokens
etc.
Simply put, if we want to be handling P sequences in parallel and we know that each sequence can have a maximum of T tokens (prompt + generated), then we want to set our KV cache size (i.e. --ctx-size) to T\*P in order to be able to handle the worst-case scenario where all P sequences fill-up the maximum T tokens.

Since llama.cpp implements a "unified" cache strategy, the KV cache size is actually shared across all sequences. This means that it's allowed to have sequences with more than T tokens as long as the sum of all tokens does not exceed P\*T.
