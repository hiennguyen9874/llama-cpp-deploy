### Qwen3-Embedding

- https://github.com/ggml-org/llama.cpp/pull/14029
- https://github.com/ggml-org/llama.cpp/issues/14234#issuecomment-2979796913
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

```bash
CUDA_VISIBLE_DEVICES=0 ./llama.cpp/build/bin/llama-server \
  -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --alias qwen3-embedding-0.6b \
  --embeddings --pooling last -ub 8192 \
  --host 0.0.0.0 --port 8081 \
  --verbose-prompt \
  --api-key llama-cpp-api-key \
  -fa on -ngl 999 --device CUDA0 \
  --parallel 4 \
  --threads 32 --threads-http -1 \
  --metrics --slots
```

```bash
curl --request POST \
    --url http://localhost:8081/v1/embeddings \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer llama-cpp-api-key" \
    --data '{"input": "Hello embeddings"}' \
    --silent
```

```
"embedding_model_name": "Qwen/Qwen3-Embedding-4B",
"max_context_tokens": 32768,
"embedding_dimension": 2560,

self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["embedding_model_name"], padding_side='left')
self.model = AutoModel.from_pretrained(CONFIG["embedding_model_name"])
self.model.to(self.device)
if self.device == "cuda":
    self.model = self.model.half()  # Convert to float16

self.max_length = CONFIG["max_context_tokens"]

# Task description from pair embedding generator
self.task_description = 'Given this project documentation, create a comprehensive embedding that focuses on project purpose and scope of work, technical details and implementation, and domain-specific information'
instruction_template = f'Instruct: {self.task_description}\nQuery:'
instruction_tokens = len(self.tokenizer.encode(instruction_template))
self.effective_max_tokens = self.max_length - instruction_tokens
```

```
"input": ["test<|endoftext|>"],
```

- To ensure embeddings follow instructions during downstream tasks, we concatenate the instruction
  and the query into a single input context, while leaving the document unchanged before processing
  with LLMs. The input format for queries is as follows:

```
{Instruction} {Query}<|endoftext|>
```
