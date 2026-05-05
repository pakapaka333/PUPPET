"""
Generate responses from the base model and the trained (DPO) model on the test dataset.

Saves one JSONL file per model under the output directory.
Generation logic is shared with 2_build_dataset/2_sample_llm_responses.py.
"""
import argparse
import importlib.util
import json
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

# Import shared generation functions from 2_build_dataset/2_sample_llm_responses.py
_sampler_path = SCRIPTS_ROOT / "2_build_dataset" / "2_sample_llm_responses.py"
_spec = importlib.util.spec_from_file_location("sample_llm_responses", _sampler_path)
_sampler_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sampler_mod)
build_prompts = _sampler_mod.build_prompts
generate = _sampler_mod.generate


def main(args):
    from datasets import load_from_disk

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scripts_root = SCRIPTS_ROOT
    input_dir = scripts_root / config["input_dir"]
    output_dir = scripts_root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading test dataset from {input_dir} ...")
    dataset = load_from_disk(str(input_dir))
    sample_df = dataset.to_pandas()

    filled_prompts = build_prompts(sample_df, config["prompt_template"])
    conversations = [[{"role": "user", "content": p}] for p in filled_prompts]

    for model_key, model_cfg in config["models"].items():
        logger.info(f"Generating responses with model: {model_key}")

        model_name = model_cfg.get("name") or model_cfg.get("base_name")
        model_kwargs = model_cfg.get("kwargs", {})
        raw_path = model_cfg.get("path")
        lora_path = str(scripts_root / raw_path) if raw_path else None

        model_outputs = generate(
            model_name, model_kwargs, conversations, config["sampling"], lora_path
        )

        result_df = sample_df.join(
            pd.DataFrame({
                "filled_prompt": filled_prompts,
                "vllm_model_output": model_outputs,
                "config": json.dumps(config),
            })
        )
        out_path = output_dir / f"{model_key}.jsonl"
        result_df.to_json(str(out_path), orient="records", lines=True, force_ascii=False)
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
