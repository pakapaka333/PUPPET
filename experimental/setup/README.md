# experimental/setup 🧪⚙️

Run `setup.sh` from the repository root to set up all external dependencies for the experimental scripts:

```
bash experimental/setup/setup.sh
```

This runs the following two scripts in order:

| Script | What it does |
|---|---|
| `scripts/setup_submodules.sh` | Adds WaterBench, dipper, and MarkLLM as submodule/subtrees and applies patches |
| `scripts/setup_eval.py` | Downloads evaluation materials into `evaluation/materials/` |

---

## `scripts/setup_submodules.sh`

Adds three external repositories and applies patches:

### WaterBench (submodule)

Added as a git submodule at `experimental/submodules/WaterBench`.

Source: [THU-KEG/WaterBench](https://github.com/THU-KEG/WaterBench) @ `8f3d779`

### ai-detection-paraphrases / dipper (subtree)

Added as a git subtree at `experimental/submodules/ai-detection-paraphrases`.

Source: [martiansideofthemoon/ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases) @ `95f3e2c`

**Patch applied (`dipper.patch`):**
- `DipperParaphraser.__init__` now accepts a `model_kwargs` argument passed to `T5ForConditionalGeneration.from_pretrained`.

### MarkLLM (subtree)

Added as a git subtree at `experimental/submodules/MarkLLM`.

Source: [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) @ `c2b773d`

**Patch applied (`markllm.patch`):**
- Avoid OOM in `EXPGumbel`: Gumbel table is generated on the fly per token instead of being pre-allocated in full.
- Support `use_chat_template` kwarg in `BaseWatermark`, `EXPGumbel`, `KSemStamp`, `KGW`, `SynthID`, and `Unigram`.
- Fix `KSemStampConfig` to prefer `gen_kwargs` over config file for `max_new_tokens`/`min_new_tokens`.
- Update `KSEMSTAMP.json` centroid path to match the subtree prefix.

**Large files removed after adding:**
- `experimental/submodules/MarkLLM/dataset/c4-train`
- `experimental/submodules/MarkLLM/watermark/xsir/dictionary/dictionary.txt`

---

## `scripts/setup_eval.py`

Downloads external files into `evaluation/materials/`. The following files are generated (not tracked by git):

### `evaluation/materials/llm_as_a_judge_prompts/ielts.txt`

An IELTS essay scoring prompt used as input to the LLM-as-a-judge evaluator.

Derived from `prompt_template` in [PON2020/IELTSWriting — gpt4o_without_example.ipynb](https://github.com/PON2020/IELTSWriting/blob/61d284bbaa7b494cf5a43775d319d46ca1999ffe/study_2_code/gpt4o_without_example.ipynb) ([Qiu et al. "Large Language Models For Second Language English Writing Assessments: An Exploratory Comparison". PACLIC2024.](https://aclanthology.org/2024.paclic-1.36/)), with the following edits:

- `{prompt_text}` → `{context}`
- `{essay_text}` → `{prediction}`
- Removed `" with justification."` from the expected response format line (score-only output)

### `evaluation/materials/utils/mage_utils.py`

Preprocessing utilities for the [MAGE](https://github.com/yafuly/MAGE) AI-text detector.

Downloaded as-is from [yafuly/MAGE — deployment/utils.py](https://github.com/yafuly/MAGE/blob/main/deployment/utils.py).

### `evaluation/materials/utils/fakepostai_utils.py`

Preprocessing utilities for the [fakespot-ai/roberta-base-ai-text-detection-v1](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1) AI-text detector.

Downloaded as-is from [fakespot-ai/roberta-base-ai-text-detection-v1 — utils.py](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/main/utils.py).
