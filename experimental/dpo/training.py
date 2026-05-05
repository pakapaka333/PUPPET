from common.utils import get_logger, load_config, save_results_as_jsonl

logger = get_logger(__name__)

import argparse
from datasets import load_dataset, load_from_disk, Dataset
import os
import pandas as pd
from peft import LoraConfig, get_peft_model

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_CONFIG_FIELDS = [
    "model/name",
    "training",
    "file/dataset",
    "file/output",
]


def __check_train_data(train_dataset:Dataset) -> bool:
    required_columns = ["prompt", "chosen", "rejected"]
    if any(col not in train_dataset.column_names for col in required_columns):
        raise ValueError(f"Training data is missing required columns: {required_columns}")
    return True


def main(args):
    # Load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)
    ckpt_path = os.path.join(
        os.environ['ROOT_DIR'],
        config['file']['output']['path'],
        os.environ['RUN_NAME'],
    )

    # Load training data
    dataset_path = config['file']['dataset']
    if os.path.isdir(dataset_path):
        all_dataset = load_from_disk(dataset_path)
        if isinstance(all_dataset, Dataset):
            all_dataset = {"train": all_dataset}
    else:
        all_dataset = load_dataset(dataset_path, token=os.environ['HF_TOKEN'])
    train_dataset =  all_dataset['train']
    __check_train_data(train_dataset)
    eval_dataset = all_dataset.get('test', None)
    if eval_dataset is not None:
        __check_train_data(eval_dataset)
    dataset_kwargs = config['training']['dataset_kwargs']
    if dataset_kwargs.get('max_train_samples', None) is not None:
        train_dataset = train_dataset.select(range(dataset_kwargs['max_train_samples']))
        logger.info(f"Reduced training dataset: {len(train_dataset)} samples")
    if (eval_dataset is not None) and (dataset_kwargs.get('max_eval_samples', None) is not None):
        eval_dataset = eval_dataset.select(range(dataset_kwargs['max_eval_samples']))
        logger.info(f"Reduced eval dataset: {len(eval_dataset)} samples")

    # Setup model
    model_name = config['model']['name']
    logger.info(f"Setting up model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **config['model']['kwargs'])
    if config['training'].get('use_lora', False):
        logger.info(f"Using LoRA.")
        lora_config = LoraConfig(**config['training']['lora_kwargs'])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info(f"Not using LoRA. Proceeding with full parameters training.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Setup trainer
    training_config = DPOConfig(
        output_dir=ckpt_path,
        report_to="wandb",
        **config['training']['dpo_kwargs']
        )
    trainer_kwargs = {
        "model": model,
        "args": training_config,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer = DPOTrainer(**trainer_kwargs)

    # Run training
    logger.info(f"Running training")
    trainer.train()

    # Save model
    logger.info(f"Saving model")
    trainer.save_state()
    trainer.save_model(ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)