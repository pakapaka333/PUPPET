> [!NOTE]
> This directory contains the code used in our experiments, provided for reproducibility purposes. <br>
> As this code has not been refactored, some parts may be difficult to set up or run. <br>
> If you wish to apply PUPPET to your own setup, please refer to the clean implementation in the [repository root](../README.md) instead. <br>
> Some internal information (e.g. absolute paths and private repository IDs) has been replaced with publishable alternatives.

<br>

---

# 🧪 Experimental Code

## Setup (first time only)

Downloads submodules and evaluation materials:

```bash
bash experimental/setup/setup.sh
```

See [`experimental/setup/README.md`](setup/README.md) for details.

---

## PUPPET <img src="../readme_assets/puppet_greeting.png" height="20"> — data creation → training → evaluation → analysis

Each script takes a YAML config as the first argument. <br>
Prepare one under the module's `configs/` directory before running.

### Data creation (`dpo_data` → `vllm_gen` → `evaluation` → `dpo_data`)

Build the preference dataset iteratively. <br>
Launch JupyterLab to run the dataset-building notebook, and check the output under `dpo_data/data/`.

```bash
bash experimental/dpo_data/run_jupyterlab.sh
```

Generate texts from the current model using vLLM. Output is saved in the directory specified in the config.

```bash
bash experimental/vllm_gen/run_generation.sh path/to/config.yaml <CUDA_VISIBLE_DEVICES>
```

Score the generated texts (task performance, detectability, etc.). Results are written to `evaluation/results/` and logs to `evaluation/logs/`. 
`CUDA_VISIBLE_DEVICES` is optional for CPU-only detectors.

```bash
bash experimental/evaluation/run_evaluation.sh path/to/config.yaml [CUDA_VISIBLE_DEVICES]
```

Repeat the three steps above until the preference dataset is complete.

### Model training (`dpo`)

Train the model via DPO. The project name and run name are read from the config and automatically set as W&B environment variables; check training progress on your W&B dashboard. Checkpoints are saved under `dpo/checkpoints/`.

```bash
bash experimental/dpo/run_training.sh path/to/config.yaml <CUDA_VISIBLE_DEVICES>
```

### Evaluation (`evaluation`)

Evaluate the trained model using the same script as above.

```bash
bash experimental/evaluation/run_evaluation.sh path/to/config.yaml [CUDA_VISIBLE_DEVICES]
```

Results are saved to `evaluation/results/`. Compare runs by inspecting the JSON files there.

### Analysis (`analysis`, `shap_analysis`, `paraphrase`)

Explore results through basic analysis, SHAP analysis or paraphrasing attack. <br>
 Notebook outputs are saved in each module's directory.

```bash
# Basic analysis
bash experimental/analysis/run_jupyterlab.sh

# SHAP analysis (script)
bash experimental/shap_analysis/run_shap_analysis.sh path/to/config.yaml <CUDA_VISIBLE_DEVICES>
bash experimental/shap_analysis/run_jupyterlab.sh

# Paraphrasing attack
bash experimental/paraphrase/run_paraphrase.sh path/to/config.yaml [CUDA_VISIBLE_DEVICES]
```

> [!WARNING]
> The SHAP analysis for LLM-as-a-Judge is expected to take about one GPU-week.

---

## Watermark baseline 💧 — generate → detect → evaluate

Each script takes a YAML config as the first argument. <br>
Prepare one under the module's `configs/` directory before running.

### Watermark generation (`watermark_gen`)

Generate watermarked texts. Output is saved in the directory specified in the config.

```bash
bash experimental/watermark_gen/run_generation.sh path/to/config.yaml <CUDA_VISIBLE_DEVICES>
```

### Watermark detection (`watermark_det`)

Detect watermarks in the generated texts. Results are saved in the directory specified in the config.

```bash
bash experimental/watermark_det/run_detection.sh path/to/config.yaml <CUDA_VISIBLE_DEVICES>
```

### Evaluation (`evaluation`)

Score the results using the same evaluation script as PUPPET. Results are saved to `evaluation/results/`.

```bash
bash experimental/evaluation/run_evaluation.sh path/to/config.yaml [CUDA_VISIBLE_DEVICES]
```
