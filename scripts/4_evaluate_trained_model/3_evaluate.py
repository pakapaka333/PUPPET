"""
Score the generated responses and print a before/after comparison table.

Reads the JSONL files produced by 2_sample_llm_responses.py, scores each with
Rouge-L (task performance) and openai-detector (detectability), then prints a
summary showing how much the trained model improved over the base model.

Scorer functions (rouge_score_waterbench, detect_openai) are also imported by
2_build_dataset/3_build_dpo_dataset.py via importlib for DPO pair construction.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Scoring functions (also imported by 2_build_dataset/3_build_dpo_dataset.py)
# ---------------------------------------------------------------------------

def rouge_score_waterbench(predictions: list, answers: list, **kwargs) -> list:
    """
    Calculate best Rouge-L for each (prediction, answer) pair.
    predictions: list[str] or list[list[str]]
    answers:     list[list[str]]
    returns:     list[float] or list[list[float]]
    """
    from rouge import Rouge

    rouge = Rouge()

    def _safe_rouge(pred: str, gt: str) -> float:
        try:
            return rouge.get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
        except Exception:
            return 0.0

    is_list = isinstance(predictions[0], list)
    all_scores = []
    for pred_or_list, gts in tqdm(
        zip(predictions, answers), total=len(predictions), desc="Rouge-L", mininterval=2.0
    ):
        pred_list = pred_or_list if is_list else [pred_or_list]
        scores = [max(_safe_rouge(p, gt) for gt in gts) for p in pred_list]
        all_scores.append(scores if is_list else scores[0])
    return all_scores


def detect_openai(predictions: list, **kwargs) -> list:
    """
    Score predictions with openai-community/roberta-large-openai-detector.
    predictions: list[str] or list[list[str]]
    returns:     list[float] or list[list[float]]
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "openai-community/roberta-large-openai-detector"
    detector_max_len = 512
    machine_class_idx = 0  # label "Fake" = AI-generated

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    detector = AutoModelForSequenceClassification.from_pretrained(
        model_name, **kwargs.get("detector_kwargs", {})
    )

    is_list = isinstance(predictions[0], list)
    all_scores = []
    with torch.no_grad():
        for pred_or_list in tqdm(predictions, desc="detect_openai", mininterval=2.0):
            pred_list = pred_or_list if is_list else [pred_or_list]
            enc = tokenizer(
                pred_list,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=detector_max_len,
            )
            enc = {k: v.to(detector.device) for k, v in enc.items()}
            logits = detector(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, machine_class_idx].detach().cpu().tolist()
            all_scores.append(probs if is_list else probs[0])
    return all_scores


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _score_df(df: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    # vllm_model_output: list[list[str]] — n predictions per sample
    model_preds: list[list[str]] = df["vllm_model_output"].apply(
        lambda x: x if isinstance(x, list) else [x]
    ).tolist()
    answers: list[list[str]] = df["outputs"].tolist()  # human answers

    # --- ROUGE-L ---------------------------------------------------------------
    # rouge_score_waterbench returns list[list[float]]: score per prediction.
    # Each score is already the best over all reference answers.
    # Average across predictions gives the per-sample ROUGE-L.
    rouge_scores: list[list[float]] = rouge_score_waterbench(model_preds, answers)
    rouge_mean = sum(sum(s) / len(s) for s in rouge_scores) / len(rouge_scores)

    # --- AUROC -----------------------------------------------------------------
    # Flatten: model predictions (label=1) and human answers (label=0).
    # Scoring is done in a single detect_openai call to load the model only once.
    flat_model: list[str] = [p for preds in model_preds for p in preds]
    flat_human: list[str] = [g for gts   in answers    for g in gts  ]

    all_det: list[float] = detect_openai(
        flat_model + flat_human, detector_kwargs={"device_map": "auto"}
    )
    n_model = len(flat_model)
    model_det = all_det[:n_model]
    human_det = all_det[n_model:]

    y_score = model_det + human_det
    y_true  = [1] * len(model_det) + [0] * len(human_det)

    auroc = roc_auc_score(y_true, y_score)

    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    valid = [(f, t) for f, t in zip(fpr_arr, tpr_arr) if f <= 0.01]
    tpr_at_1fpr = valid[-1][1] if valid else 0.0

    return {
        "rouge_mean":   rouge_mean,
        "detect_auroc": auroc,
        "tpr_at_1fpr":  tpr_at_1fpr,
    }


def _print_results(results: dict) -> None:
    W_LABEL = 32
    W_VAL   = 10
    W_DELTA = 14
    W_TOTAL = W_LABEL + W_VAL + W_VAL + W_DELTA + 3  # +3 for column spaces

    metrics = [
        ("Rouge-L",                     "rouge_mean"),
        ("OpenAI detector AUROC",       "detect_auroc"),
        ("OpenAI detector TPR@1%FPR",   "tpr_at_1fpr"),
    ]

    print("\n" + "=" * W_TOTAL)
    print(f"{'Metric':<{W_LABEL}} {'base':>{W_VAL}} {'trained':>{W_VAL}} {'Δ':>{W_DELTA}}")
    print("-" * W_TOTAL)
    for label, key in metrics:
        base_val    = results.get("base",    {}).get(key, float("nan"))
        trained_val = results.get("trained", {}).get(key, float("nan"))
        delta       = trained_val - base_val
        delta_str   = f"{delta:+.4f}"
        print(
            f"{label:<{W_LABEL}}"
            f" {base_val:{W_VAL}.4f}"
            f" {trained_val:{W_VAL}.4f}"
            f" {delta_str:>{W_DELTA}}"
        )
    print("=" * W_TOTAL + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    scripts_root = SCRIPTS_ROOT
    responses_dir = scripts_root / config["responses_dir"]
    output_dir    = scripts_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = config.get("model_keys", ["base", "trained"])
    results: dict[str, dict] = {}

    for key in model_keys:
        path = responses_dir / f"{key}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Response file not found: {path}")
        logger.info(f"Scoring {key} model responses ...")
        df = pd.read_json(str(path), orient="records", lines=True)
        results[key] = _score_df(df)

        out_path = output_dir / f"{key}_scored.jsonl"
        df.to_json(str(out_path), orient="records", lines=True, force_ascii=False)

    _print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
