# NanoChat Fine-Tuning Guide: Training with Your Own Data

This guide walks you through fine-tuning a pretrained NanoChat model on your own conversational data using Supervised Fine-Tuning (SFT).

---

## Overview

NanoChat's fine-tuning pipeline takes a pretrained base model and teaches it to follow instructions and hold conversations. The process uses the `scripts/chat_sft.py` script with data loaded through task classes defined in `tasks/`.

The simplest way to inject your own data is through the **CustomJSON** task (`tasks/customjson.py`), which reads conversations from a JSONL file.

---

## Data Format

### JSONL File Structure

Your data must be a `.jsonl` (JSON Lines) file where **each line** is a standalone JSON array of messages representing one conversation.

**Each line** is a JSON array (not a JSON object — just the array of messages directly):

```jsonl
[{"role":"user","content":"What is photosynthesis?"},{"role":"assistant","content":"Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."}]
[{"role":"user","content":"Hi, can you help me?"},{"role":"assistant","content":"Of course! What do you need help with?"},{"role":"user","content":"How do I sort a list in Python?"},{"role":"assistant","content":"You can use the built-in sorted() function or the .sort() method. For example: sorted([3, 1, 2]) returns [1, 2, 3]."}]
```

### Message Format

Each message in the array is a JSON object with two required fields:

| Field     | Type   | Description                        |
|-----------|--------|------------------------------------|
| `role`    | string | Either `"user"` or `"assistant"`   |
| `content` | string | The text content of the message    |

### Strict Rules

1. **Alternating roles**: Messages must strictly alternate `user, assistant, user, assistant, ...`
2. **Start with user**: The first message must have `role: "user"`
3. **Minimum 2 messages**: Each conversation needs at least one user message and one assistant response
4. **Even or odd count**: Conversations can end on either a user or assistant message, but typically end with an assistant message
5. **No system messages**: Unlike SmolTalk (which supports an optional system message), CustomJSON expects strictly alternating user/assistant messages with no system role
6. **String content only**: The `content` field must be a plain string

### Example: Single-Turn Conversation

```json
[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"The capital of France is Paris."}]
```

### Example: Multi-Turn Conversation

```json
[{"role":"user","content":"I want to learn about neural networks."},{"role":"assistant","content":"Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information. What aspect interests you most?"},{"role":"user","content":"How does backpropagation work?"},{"role":"assistant","content":"Backpropagation is the algorithm used to train neural networks. It works in two phases:\n\n1. Forward pass: Input flows through the network to produce an output.\n2. Backward pass: The error (difference between predicted and actual output) is propagated backwards through the network, and each weight is adjusted proportionally to its contribution to the error.\n\nThis uses the chain rule from calculus to compute gradients efficiently."}]
```

### Example: Full JSONL File

Save this as `my_data.jsonl`:

```jsonl
[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"2 + 2 = 4."}]
[{"role":"user","content":"Tell me a joke"},{"role":"assistant","content":"Why don't scientists trust atoms? Because they make up everything!"}]
[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi there! How can I help you today?"},{"role":"user","content":"What's the weather like?"},{"role":"assistant","content":"I don't have access to real-time weather data, but I'd recommend checking a weather service for your area."}]
```

---

## Advanced Data Format: Tool Use (Python REPL)

For conversations that include tool use (Python code execution), the assistant's `content` field can be a **list of parts** instead of a plain string. This is how NanoChat learns to use its built-in Python calculator/REPL.

Each part is an object with `type` and `text` fields:

| Part Type        | Description                                   | Supervised? |
|------------------|-----------------------------------------------|-------------|
| `text`           | Regular assistant text                        | Yes         |
| `python`         | Python code the assistant writes to execute   | Yes         |
| `python_output`  | Output returned by the Python execution       | No          |

The model is trained to **generate** `text` and `python` parts, but `python_output` tokens are masked from the loss (since at inference time, those come from actual code execution).

### Tool Use Example

```json
[
  {"role": "user", "content": "What is 1547 * 382?"},
  {"role": "assistant", "content": [
    {"type": "text", "text": "Let me calculate that.\n"},
    {"type": "python", "text": "1547 * 382"},
    {"type": "python_output", "text": "590954"},
    {"type": "text", "text": "\n1547 * 382 = 590,954."}
  ]}
]
```

---

## How Tokenization Works

Understanding how your data gets tokenized helps you write better training data.

NanoChat uses special tokens to delimit conversation structure:

```
<|bos|>  <|user_start|> user message <|user_end|>  <|assistant_start|> assistant response <|assistant_end|>
```

Key details about the loss mask:
- **User messages**: All tokens (including `<|user_start|>`, `<|user_end|>`) have mask=0 (not trained on)
- **Assistant messages**: Content tokens have mask=1 (trained on), and `<|assistant_end|>` has mask=1
- **`<|assistant_start|>`**: Has mask=0 (not trained on)
- **Tool calls**: `<|python_start|>`, code, `<|python_end|>` all have mask=1 (trained on)
- **Tool outputs**: `<|output_start|>`, output text, `<|output_end|>` all have mask=0 (not trained on)

The model only learns to predict what the assistant should say — it never trains on user prompts or tool outputs.

### Context Length

Conversations are truncated to `max_seq_len` tokens (default: 2048, inherited from pretrained checkpoint). Keep conversations concise enough to fit within this limit.

### Conversation Packing

The SFT dataloader uses a **best-fit packing** strategy: multiple short conversations are packed into a single training row to maximize GPU utilization. Each conversation starts with a `<|bos|>` token. If no conversation fits the remaining space in a row, it is padded (never truncated mid-conversation).

---

## Step-by-Step: Fine-Tuning with Custom Data

### Step 1: Prepare Your Data

Create your JSONL file following the format above. Save it somewhere accessible, e.g.:

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
cp my_data.jsonl $NANOCHAT_BASE_DIR/my_data.jsonl
```

### Step 2: Ensure You Have a Pretrained Base Model

You need a pretrained checkpoint. Either train one from scratch:

```bash
# Quick small model (~5-10 min on 8xH100)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=12 --run="d12" --model-tag="d12"
```

Or use a full-scale model from `runs/speedrun.sh`.

### Step 3: Modify the SFT Script

Edit `scripts/chat_sft.py` to include your custom data in the training mixture. Find the `train_tasks` list (around line 165) and add your dataset:

```python
# Add your custom data to the training mixture
my_data_filepath = os.path.join(base_dir, "my_data.jsonl")
train_tasks = [
    SmolTalk(split="train"),
    CustomJSON(filepath=my_data_filepath),        # your data (1 epoch)
    CustomJSON(filepath=my_data_filepath),        # your data (2 epochs for emphasis)
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),
    *[MMLU(subset="auxiliary_train", split="train") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    SimpleSpelling(size=200000, split="train"),
    SpellingBee(size=80000, split="train"),
]
```

**Tip**: Pass the same `CustomJSON(filepath=...)` multiple times to oversample your data (train on it for more epochs). This is useful when your dataset is small relative to the other tasks.

### Step 4: Run Fine-Tuning

Single GPU:

```bash
python -m scripts.chat_sft --model-tag="d12"
```

Multi-GPU with DDP:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --device-batch-size=16 --run=my_sft_run
```

### Step 5: Evaluate

```bash
# Run benchmarks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Chat with your model
python -m scripts.chat_cli -p "Hello!"

# Or use the web UI
python -m scripts.chat_web
```

---

## Fine-Tuning Only on Your Data

If you want to fine-tune exclusively on your own dataset (no SmolTalk, MMLU, etc.), replace the entire `train_tasks` list:

```python
my_data_filepath = os.path.join(base_dir, "my_data.jsonl")
train_tasks = [
    CustomJSON(filepath=my_data_filepath),
]
```

And set up an appropriate validation set:

```python
val_dataset = TaskMixture([
    CustomJSON(filepath=os.path.join(base_dir, "my_data_val.jsonl")),
])
```

Be aware that training only on a narrow dataset may cause the model to lose general capabilities (catastrophic forgetting). Mixing with SmolTalk helps preserve general conversation ability.

---

## Creating a Custom Task Class

For more control over data loading, you can write your own Task class. Create a new file in `tasks/` following this pattern:

```python
# tasks/my_task.py
from tasks.common import Task

class MyTask(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        # Load your data here
        self.data = self._load_data(split)

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        # Must return a dict with a "messages" key
        # containing a list of {"role": ..., "content": ...} dicts
        return {"messages": self.data[index]}
```

Then import and use it in `scripts/chat_sft.py`:

```python
from tasks.my_task import MyTask

train_tasks = [
    SmolTalk(split="train"),
    MyTask(split="train"),
    # ...
]
```

Each `get_example()` must return a conversation dict:

```python
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},  # string or list of parts
    ]
}
```

---

## Generating Synthetic Training Data

NanoChat includes a synthetic data generator (`dev/gen_synthetic_data.py`) that uses an LLM API to create diverse conversations. This is useful for:

- Teaching the model about its own identity
- Generating domain-specific Q&A pairs at scale
- Creating diverse conversation styles and personas

The generator injects diversity across multiple dimensions:
- **Topics**: What the conversation covers
- **Personas**: Who is asking (beginner, expert, skeptic, etc.)
- **Dynamics**: Conversation shape (short Q&A, deep technical, skeptical arc, etc.)
- **First messages**: Greeting style variation

To use it:

```bash
# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Generate 1000 conversations
python dev/gen_synthetic_data.py --num 1000 --workers 4 --output my_data.jsonl
```

You can adapt the prompt template, topics, and personas in that script for your own domain.

---

## Key Training Parameters

| Parameter              | Default        | Description                                                |
|------------------------|----------------|------------------------------------------------------------|
| `--num-iterations`     | -1 (full epoch)| Number of optimization steps                               |
| `--max-seq-len`        | inherited      | Max context length (tokens per sequence)                   |
| `--device-batch-size`  | inherited      | Per-GPU batch size                                         |
| `--total-batch-size`   | inherited      | Total batch size in tokens across all GPUs                 |
| `--init-lr-frac`       | 0.8            | Initial LR as fraction of base pretrain LR                 |
| `--warmup-ratio`       | 0.0            | Fraction of training for LR warmup                         |
| `--warmdown-ratio`     | 0.5            | Fraction of training for LR cooldown                       |
| `--eval-every`         | 200            | Evaluate validation BPB every N steps                      |
| `--chatcore-every`     | 200            | Evaluate ChatCORE benchmark every N steps                  |
| `--mmlu-epochs`        | 3              | Epochs of MMLU in training mixture                         |
| `--gsm8k-epochs`       | 4              | Epochs of GSM8K in training mixture                        |
| `--load-optimizer`     | 1              | Warm-start optimizer from pretrained checkpoint (0 or 1)   |

---

## Tips for Good Results

1. **Data quality matters more than quantity.** A few thousand high-quality, well-structured conversations beat millions of noisy ones at this model scale.

2. **Oversample small datasets.** If your custom data is small (e.g., 500 conversations) relative to SmolTalk (460K), include it multiple times in `train_tasks` so the model sees it enough.

3. **Keep conversations within context length.** The default is 2048 tokens. Long conversations get truncated, losing the end of the conversation.

4. **Mix with general data to avoid catastrophic forgetting.** Keep SmolTalk in the mix unless you intentionally want a narrow specialist model.

5. **Validate your JSONL before training.** A malformed line will crash the loader. Quick validation:

   ```python
   import json
   with open("my_data.jsonl") as f:
       for i, line in enumerate(f):
           msgs = json.loads(line)
           assert isinstance(msgs, list) and len(msgs) >= 2
           for j, m in enumerate(msgs):
               assert m["role"] == ("user" if j % 2 == 0 else "assistant")
               assert isinstance(m["content"], str)
   print(f"Validated {i+1} conversations successfully")
   ```

6. **Use wandb for monitoring.** Set `--run=my_experiment` to log training curves.

7. **Start small.** Use `--depth=12` and `--num-iterations=500` for quick experiments before committing to a full training run.
