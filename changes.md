First pass changes. (Generated with Claude)

# Changes to train_and_publish.py

## Summary
Replaced toy dummy data with real multi-task training across three datasets targeting
IFEval >= 45%, GSM8K >= 50%, and HumanEval >= 30%.

---

## 1. Replaced DEMO_CONVERSATIONS with real datasets

**What changed:** Removed the 8 hardcoded dummy conversations. Added three dataset
loaders: `load_gsm8k()`, `load_tulu()`, and `load_opencode()`.

**Why:** The toy data was just for verifying the pipeline. Real training requires
domain-specific examples that teach the model the skills being evaluated.

---

## 2. GSM8K — load_gsm8k()

**What changed:** Loads all 7,473 examples from the `openai/gsm8k` train split.
Converts each example from `{question, answer}` fields into conversation format
`[{role: user, content: question}, {role: assistant, content: answer}]`.

**Why:** GSM8K is small enough to use entirely. The conversion is necessary because
`conversation_to_datum` expects a list of role/content dicts, not raw fields.

---

## 3. Tulu-3 — load_tulu()

**What changed:** Streams 7,000 examples from `allenai/tulu-3-sft-mixture`.
No conversion needed since Tulu is already in messages format.

**Why:** Tulu-3 is a broad instruction-following mixture (939k examples) designed
to improve general instruction compliance — directly targeting IFEval. We sample
7k to balance it against the other two tasks. Streaming avoids downloading the
full 939k dataset.

---

## 4. OpenCodeInstruct — load_opencode()

**What changed:** Streams up to 7,000 examples from `nvidia/OpenCodeInstruct`,
filtering to only include examples with `average_test_score >= 0.8`. Converts
from `{input, output}` fields into conversation format.

**Why:** OpenCodeInstruct is 5M examples so streaming + filtering is essential.
The quality filter ensures we only train on code examples where the generated
solution actually passes most unit tests, avoiding noisy or incorrect training signal.

---

## 5. Equal data mixing and shuffling

**What changed:** All three datasets are concatenated (~21k total) and shuffled
with a fixed random seed (42) before tokenization.

**Why:** Equal representation prevents the model from overfitting to one task at
the expense of others. Shuffling ensures batches contain a mix of task types,
which stabilizes multi-task gradient updates. Fixed seed makes runs reproducible.

---

## 6. num_steps: 10 → 500

**What changed:** Default `--num_steps` increased from 10 to 500.

**Why:** 10 steps is only enough to verify the pipeline. With ~21k examples and
batch size 4, 500 steps covers ~2,000 examples — enough for meaningful learning
while keeping runtime reasonable (~10-20 minutes on Tinker).

---

## 7. LoRA rank: 32 → 64

**What changed:** Default `--rank` increased from 32 to 64.

**Why:** Higher rank gives the LoRA adapter more capacity to learn across three
distinct tasks simultaneously. Rank 64 is a good balance between expressiveness
and training stability for a 3B model.

---

## 8. max_length: 512 → 1024

**What changed:** `max_length` increased from 512 to 1024 tokens.

**Why:** GSM8K answers include chain-of-thought reasoning steps, and code examples
can be long. 512 tokens truncates many examples mid-answer, which corrupts the
training signal. 1024 captures most examples fully.

---

## 9. checkpoint_name: "demo" → "multitask-v1"

**What changed:** Default checkpoint name updated.

**Why:** Descriptive naming makes it easier to track experiments on the Tinker
dashboard when running multiple training runs.

---

## 10. Loss logging: every step → every 50 steps

**What changed:** Loss is now printed at step 1 and every 50 steps thereafter.

**Why:** With 500 steps, printing every step produces 500 lines of output.
Logging every 50 keeps the output readable while still showing the loss curve.

---

## Hyperparameters (unchanged from toy example)

- `batch_size`: 4 — kept the same, sufficient for 3B model
- `lr`: 1e-4 — standard for LoRA fine-tuning
- Adam betas: 0.9 / 0.95, eps: 1e-8 — standard values, no reason to change