import argparse
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent

# Import scoring functions from 4_evaluate_trained_model/3_evaluate.py
_eval_path = SCRIPTS_ROOT / "4_evaluate_trained_model" / "3_evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate", _eval_path)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
rouge_score_waterbench = _eval_mod.rouge_score_waterbench
detect_openai = _eval_mod.detect_openai


def _standardize(values: list) -> list:
    m = sum(values) / len(values)
    s = (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5
    return [(v - m) / (s or 1.0) for v in values]


def _weighted_scores(row: pd.Series, score_weights: dict) -> list[float]:
    """Compute per-candidate weighted sum of standardized scores for one sample."""
    raw = {name: row[name] for name in score_weights}
    std = {name: _standardize(vals) for name, vals in raw.items()}
    k = len(row["vllm_model_output"])
    return [
        sum(score_weights[name] * std[name][i] for name in score_weights)
        for i in range(k)
    ]


def _strategy_best_and_worst(gen_df: pd.DataFrame, score_weights: dict) -> pd.DataFrame:
    """Pair the highest-scoring candidate (chosen) with the lowest-scoring one (rejected)."""
    rows = []
    for _, row in gen_df.iterrows():
        preds = row["vllm_model_output"]
        weighted = _weighted_scores(row, score_weights)

        chosen_idx  = weighted.index(max(weighted))
        rejected_idx = weighted.index(min(weighted))

        rows.append({
            **row.to_dict(),
            "chosen":         preds[chosen_idx],
            "rejected":       preds[rejected_idx],
            "chosen_score":   weighted[chosen_idx],
            "rejected_score": weighted[rejected_idx],
        })
    return pd.DataFrame(rows)


# Add new strategies here.
_PAIR_STRATEGIES = {
    "best_and_worst": _strategy_best_and_worst,
}


def _make_dpo_pairs(
    gen_df: pd.DataFrame, score_weights: dict, pair_strategy: str
) -> pd.DataFrame:
    """Dispatch to the requested pairing strategy."""
    if pair_strategy not in _PAIR_STRATEGIES:
        raise ValueError(
            f"Unknown pair_strategy '{pair_strategy}'. "
            f"Available: {list(_PAIR_STRATEGIES)}"
        )
    return _PAIR_STRATEGIES[pair_strategy](gen_df, score_weights)


def _to_conversation(role: str, text: str) -> list:
    return [{"role": role, "content": text}]


def main(args):
    from datasets import Dataset

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scripts_root = SCRIPTS_ROOT
    responses_path = scripts_root / config["input"]["responses_path"]
    output_dir = scripts_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    score_weights: dict = config["score_weights"]
    pair_strategy: str  = config.get("pair_strategy", "best_and_worst")

    # Load data (base dataset is already joined into the responses JSONL)
    logger.info(f"Loading LLM responses from {responses_path} ...")
    gen_df = pd.read_json(str(responses_path), orient="records", lines=True)

    answers = gen_df["outputs"].tolist()
    predictions = gen_df["vllm_model_output"].tolist()

    # Score
    # If you want to use other metrics, you should modify the following code.
    logger.info("Scoring: Rouge-L ...")
    rouge_scores = rouge_score_waterbench(predictions, answers)
    gen_df["rouge_score_waterbench"] = rouge_scores

    logger.info("Scoring: OpenAI detector ...")
    detect_scores = detect_openai(predictions, detector_kwargs={"device_map": "auto"})
    gen_df["detect_openai"] = detect_scores

    # Build DPO pairs
    logger.info(f"Building DPO pairs (strategy='{pair_strategy}') ...")
    pair_df = _make_dpo_pairs(gen_df, score_weights, pair_strategy)

    # Convert to DPO conversation format
    pair_df["prompt"] = pair_df["filled_prompt"].apply(
        lambda p: _to_conversation("user", p)
    )
    pair_df["chosen"] = pair_df["chosen"].apply(
        lambda t: _to_conversation("assistant", t)
    )
    pair_df["rejected"] = pair_df["rejected"].apply(
        lambda t: _to_conversation("assistant", t)
    )

    # Keep only DPO-required columns
    keep_cols = ["prompt", "chosen", "rejected", "chosen_score", "rejected_score"]
    dpo_df = pair_df[keep_cols]

    # Save as HuggingFace Dataset
    dpo_ds = Dataset.from_pandas(dpo_df, preserve_index=False)
    dpo_ds.save_to_disk(str(output_dir))
    logger.info(f"Saved DPO dataset ({len(dpo_ds):,} samples) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
