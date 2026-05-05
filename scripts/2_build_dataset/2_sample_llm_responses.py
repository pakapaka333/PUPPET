import argparse
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


def build_prompts(sample_df: pd.DataFrame, prompt_template: str) -> list[str]:
    """Fill prompt_template with each row of sample_df and return a list of strings."""
    return [
        prompt_template.format(context=row.get("context", ""), input=row["input"])
        for _, row in sample_df.iterrows()
    ]


def generate(
    model_name: str,
    model_kwargs: dict,
    conversations: list,
    sampling_config: dict,
    lora_path: str | None = None,
) -> list[list[str]]:
    """
    Run vLLM generation and return list[list[str]] (one inner list per sample).

    Args:
        model_name:     HuggingFace model ID or local path.
        model_kwargs:   Extra kwargs forwarded to vllm.LLM (e.g. dtype, tensor_parallel_size).
        conversations:  list of chat messages in OpenAI format.
        sampling_config: dict forwarded directly to vllm.SamplingParams (n, min_tokens, max_tokens, …).
        lora_path:      Optional path to a LoRA adapter directory.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    use_lora = lora_path is not None
    llm = LLM(model=model_name, enable_lora=use_lora, **model_kwargs)
    sampling_params = SamplingParams(**sampling_config)

    if use_lora:
        lora_req = LoRARequest("dpo", 1, lora_path)
        outputs = llm.chat(conversations, sampling_params, lora_request=lora_req)
    else:
        outputs = llm.chat(conversations, sampling_params)

    return [[cand.text for cand in out.outputs] for out in outputs]


def main(args):
    from datasets import load_from_disk

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scripts_root = SCRIPTS_ROOT
    input_dir = scripts_root / config["input_dir"]
    output_path = scripts_root / config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base dataset from {input_dir} ...")
    dataset = load_from_disk(str(input_dir))
    sample_df = dataset.to_pandas()

    n = config["sampling"].get("n")
    if n is None:
        raise ValueError("`n` (number of responses per sample) is not set in the config file.")

    filled_prompts = build_prompts(sample_df, config["prompt_template"])
    conversations = [[{"role": "user", "content": p}] for p in filled_prompts]

    logger.info(f"Loading model {config['model']['name']} ...")
    logger.info(f"Generating responses: {len(conversations)} samples × {n} per sample ...")
    model_outputs = generate(
        config["model"]["name"],
        config["model"].get("kwargs", {}),
        conversations,
        config["sampling"],
    )

    result_df = sample_df.join(
        pd.DataFrame({
            "filled_prompt": filled_prompts,
            "vllm_model_output": model_outputs,
            "config": json.dumps(config),
        })
    )
    result_df.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
