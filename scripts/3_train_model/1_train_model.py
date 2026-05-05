import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent


def _check_train_data(dataset) -> None:
    required = ["prompt", "chosen", "rejected"]
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")


def main(args):
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scripts_root = SCRIPTS_ROOT
    dataset_dir = scripts_root / config["dataset_dir"]
    output_dir = scripts_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir} ...")
    from datasets import Dataset as HFDataset

    all_dataset = load_from_disk(str(dataset_dir))
    if isinstance(all_dataset, HFDataset):
        train_dataset = all_dataset
        eval_dataset = None
    else:
        # DatasetDict
        train_dataset = all_dataset["train"]
        eval_dataset = all_dataset.get("test", None)

    _check_train_data(train_dataset)
    if eval_dataset is not None:
        _check_train_data(eval_dataset)

    dataset_kwargs = config["training"].get("dataset_kwargs", {})
    if dataset_kwargs.get("max_train_samples"):
        train_dataset = train_dataset.select(range(dataset_kwargs["max_train_samples"]))
    if eval_dataset is not None and dataset_kwargs.get("max_eval_samples"):
        eval_dataset = eval_dataset.select(range(dataset_kwargs["max_eval_samples"]))

    # Setup model
    model_name = config["model"]["name"]
    logger.info(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **config["model"]["kwargs"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if config["training"].get("use_lora", False):
        logger.info("Applying LoRA ...")
        lora_config = LoraConfig(**config["training"].get("lora_kwargs", {}))
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # W&B: use "none" when wandb key is null/absent
    report_to = "wandb" if config.get("wandb") else "none"

    training_config = DPOConfig(
        output_dir=str(output_dir),
        report_to=report_to,
        **config["training"]["dpo_kwargs"],
    )
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Training ...")
    trainer.train()

    logger.info(f"Saving model to {output_dir} ...")
    trainer.save_state()
    trainer.save_model(str(output_dir))
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
