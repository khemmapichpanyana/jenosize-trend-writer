---
base_model: unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags: [lora, qlora, sft, unsloth, trl]
---

# jeno-lora-v2

A LoRA adapter for **Qwen3-4B-Instruct-2507** that writes business trend and
future-ideas articles in the house style of Jenosize Ideas. It is the adapter
served in production as `jeno-lora`.

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (trained on Unsloth's 4-bit build) |
| Method | QLoRA SFT (Unsloth + TRL), assistant-turn loss only |
| LoRA | r = 16, alpha = 16, dropout 0, all attention and MLP projections (q, k, v, o, gate, up, down) |
| Data | 92 training articles from the v2 dataset (13 held out), fingerprint `69f0b4147fd8` |
| Schedule | 1 epoch, 12 optimizer steps, learning rate 1e-4, linear decay |
| Hardware | 1× NVIDIA L4 on Modal, about 5.5 GB peak GPU memory |
| Loss | 2.71 at step 1 → 1.98 at step 12 (epoch mean 2.28) |

**Serve with vLLM:**

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-lora \
  --lora-modules jeno-lora=./evidence/weights/jeno-lora-v2
```

Prompts must use the builder in `ai-services/app/services/prompt.py`, the same
code that rendered the training data. Evidence and evaluation results are in
[`../../README.md`](../../README.md).
