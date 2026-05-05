import argparse
import logging
import random
from pathlib import Path

from datasets import Dataset, load_dataset
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent


def build_eli5_dataset(seed: int = 42, num_samples: int = 5000, start_index: int = 0) -> Dataset:
    """
    Download and preprocess ELI-5 from Hello-SimpleAI/HC3.

    Preprocessing steps follow the WaterBench pipeline:
    - Filter samples whose average human answer length exceeds 300 words
    - Sort by average human answer length (descending)
    - Randomly shuffle with the given seed and select `num_samples` starting at `start_index`

    Returns a HuggingFace Dataset with columns compatible with the ELI-5 prompt template.
    """
    import nltk
    from nltk.tokenize import word_tokenize

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    logger.info("Loading ELI-5 dataset from Hello-SimpleAI/HC3 ...")
    ds = load_dataset("Hello-SimpleAI/HC3", "reddit_eli5", split="train")

    # The filtering and sorting steps are based on WaterBench/process/download.py
    # (https://github.com/THU-KEG/WaterBench/blob/8f3d779d66518a7b90ce1aad1fabaeb13cfca548/process/download.py#L127-L131).

    # Filter: keep samples whose average human answer length is <= 300 words
    def _add_output_length(sample):
        sample["output_length"] = sum(
            len(word_tokenize(a)) for a in sample["human_answers"]
        ) / len(sample["human_answers"])
        return sample

    logger.info("Filtering by average human answer length (<= 300 words) ...")
    ds = ds.map(_add_output_length)
    before = len(ds)
    ds = ds.filter(lambda x: x["output_length"] <= 300)
    logger.info(f"Removed {before - len(ds):,} samples; {len(ds):,} remaining.")

    # Sort descending by output_length (matches WaterBench preprocessing)
    ds = ds.sort("output_length", reverse=True)

    # Random shuffle then select slice [start_index : start_index + num_samples]
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    selected = indices[start_index : start_index + num_samples]
    ds = ds.select(selected)
    logger.info(
        f"Selected {len(ds):,} samples (seed={seed}, start={start_index}, n={num_samples})."
    )

    # Add required columns
    def _add_columns(sample):
        from nltk.tokenize import word_tokenize as _wt
        sample["input_length"] = len(_wt(sample["question"]))
        sample["length"] = sample["input_length"] + sample["output_length"]
        sample["all_classes"] = "null"
        sample["language"] = "en"
        sample["dataset"] = "longform_qa"
        sample["context"] = ""
        return sample

    ds = ds.map(_add_columns)

    # Rename / remove columns
    ds = ds.rename_columns({"id": "_id", "question": "input", "human_answers": "outputs"})
    ds = ds.remove_columns(["chatgpt_answers"])
    return ds


def build_dataset(seed: int = 42, num_samples: int = 5000, start_index: int = 0) -> Dataset:
    """
    Entry point for dataset construction used by both the training and evaluation pipelines.

    To use your own dataset, implement a function with the same signature and replace
    the call below. The returned Dataset must contain at least the following columns:

        input   (str)       — question / prompt text
        context (str)       — additional context prepended to the prompt (empty string if none)
        outputs (list[str]) — human-written reference answers
                              (used for ROUGE-L scoring and AUROC computation)

    Example:
        # def build_custom_dataset(seed, num_samples, start_index):
        #     ds = load_dataset("your/dataset")["train"]
        #     ds = ds.rename_columns({"question": "input", "answers": "outputs"})
        #     ds = ds.map(lambda x: {"context": ""})
        #     indices = list(range(len(ds)))
        #     random.seed(seed)
        #     random.shuffle(indices)
        #     return ds.select(indices[start_index: start_index + num_samples])
        #
        # return build_custom_dataset(seed=seed, num_samples=num_samples, start_index=start_index)
    """
    return build_eli5_dataset(seed=seed, num_samples=num_samples, start_index=start_index)


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = SCRIPTS_ROOT / config["output_dir"]
    ds = build_dataset(
        seed=config.get("seed", 42),
        num_samples=config.get("num_samples", 5000),
        start_index=config.get("start_index", 0),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving dataset to {output_dir} ...")
    ds.save_to_disk(str(output_dir))
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
