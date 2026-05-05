from common.utils import get_logger, load_config, save_results_as_jsonl, load_existing_results, BenchmarkManager

logger = get_logger(__name__)

import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

REQUIRED_CONFIG_FIELDS = [
    "model/name",
    "watermark",
    "watermark/active",
    "generation",
    "file/root_dir",
    "file/input/benchmark",
    "file/input/dataset",
    "file/output/path",
]


def main(args):
    # Load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)
    config_dumped = json.dumps(config)
    current_server = os.environ.get('SERVER_NAME', 'Unknown')

    # Load samples
    bm = BenchmarkManager()
    sample_df, _, filled_prompts = bm.get_samples_and_conversations(logger, config, return_filled_prompts=True)
    idx_sorted = sorted(
        range(len(filled_prompts)), 
        key=lambda i: len(filled_prompts[i]), reverse=True
        )
    model_inputs_sorted = [filled_prompts[i] for i in idx_sorted]
    min_length, max_length = bm.get_min_max_lengths(config['file']['input']['dataset'])

    # Setup model
    # ref: https://github.com/redwyd/SymMark/blob/f68a52ae4ad34632222aadcc96770af84b4c51ae/main.py#L74-L86
    model_name = config['model']['name']
    logger.info(f"Setting up model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **config['model']['kwargs']).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    merged_transformers_config_params = {
        "min_new_tokens": min_length,
        "max_new_tokens": max_length,
        "do_sample": True,
        **config['model']['transformers_config_kawrgs'],
    }

    vocab_size_cands = {
        "tokenizer_length": len(tokenizer),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_input_embeddings_shape": model.get_input_embeddings().weight.shape[0],
    }
    vocab_size_key = config['model'].get("vocab_size_key", "model_input_embeddings_shape")
    if len(set(vocab_size_cands.values())) != 1:
        logger.warning(
            "Multiple candidates for vocab size found: "
            f"{vocab_size_cands}. "
            f"Using {vocab_size_cands[vocab_size_key]} ({vocab_size_key}) as the vocab size."
        )
    vocab_size = vocab_size_cands[vocab_size_key]
    vocab_size_dumped = json.dumps({"key": vocab_size_key, "size": vocab_size})

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        pad_token_id=tokenizer.eos_token_id,
        **merged_transformers_config_params,
    )

    # Setup watermark processor
    logger.info(f"Setting up watermark processor")
    algorithm_config = config['watermark'].get(
        'algorithm_config',
        os.path.join(
            os.environ['ROOT_DIR'],
            f"submodules/MarkLLM/config/{config['watermark']['algorithm_name']}.json",
        ),
    )
    watermark = AutoWatermark.load(
        algorithm_name=config['watermark']['algorithm_name'],
        algorithm_config=algorithm_config,
        transformers_config=transformers_config,
    )

    # Run generations
    # MarkLLM only supports generation of one prompt at a time.
    # ref: https://github.com/THU-BPM/MarkLLM/blob/112fa7419b115efc82f229a59a86fb14c5528c80/watermark/kgw/kgw.py#L233
    logger.info(f"Running generations")
    model_outputs_text_all = [[] for _ in range(len(model_inputs_sorted))]
    n_generations_per_sample = config['generation']['n_generations_per_sample']

    ## Resume generation from existing results
    existing_results = load_existing_results(logger, config)
    generation_start_index = 0
    if existing_results is None:
        logger.info(f"No existing results found. Start the generation from the scratch.")
    else:
        logger.info(f"Found existing results. Try to resume the generation from the existing results.")

        loaded_filled_prompts = existing_results['filled_prompt'].tolist()
        loaded_config = existing_results['config'].tolist()[0]
        loaded_server = existing_results['server'].tolist()[0] if 'server' in existing_results.columns else "Unknown"

        loaded_model_inputs_sorted = [loaded_filled_prompts[i] for i in idx_sorted]

        is_same_config = (loaded_config == config_dumped)
        are_same_model_inputs = all(loaded_inp == inp for loaded_inp, inp in zip(loaded_model_inputs_sorted, model_inputs_sorted))
        is_same_server = (loaded_server == "Unknown") or (loaded_server == current_server)

        if not is_same_config:
            raise ValueError(f"The existing results are not for the same config of the current run, so we cannot resume the generation from the existing results.")
        if not are_same_model_inputs:
            raise ValueError(f"The existing results are not for the same model inputs of the current run, so we cannot resume the generation from the existing results.")
        if not is_same_server:
            raise ValueError(f"The loaded server {loaded_server} is not the same as the current server {os.environ.get('SERVER_NAME', 'Unknown')}, so we cannot resume the generation from the existing results.")
        if loaded_server == "Unknown":
            logger.warning(f"The loaded server is `unknown`. Please confirm the server you used to generate the existing results.")

        model_inputs_sorted = loaded_model_inputs_sorted
        model_outputs_text_all = existing_results['watermarked_model_output'].tolist()

        for j in range(len(model_outputs_text_all)):
            model_outputs_texts = model_outputs_text_all[idx_sorted[j]]
            if not isinstance(model_outputs_texts, list) or \
                (len(model_outputs_texts)>0 and not isinstance(model_outputs_texts[0], str)):
                raise ValueError("The existing results are not for the valid format, so we cannot resume the generation from the existing results.")
            if model_outputs_texts == []:
                generation_start_index = j
                break
        if j == len(model_outputs_text_all)-1:
            logger.info(f"All the existing results are already generated. No need to resume the generation.")
            return
        
        logger.info(f"Resuming the generation from the existing results. Starting from the {generation_start_index}-th result.")
        os.environ["RUN_NAME"] = os.environ["RUN_NAME"] + f"_resume_{generation_start_index}"

    ## Generate new generations
    with torch.inference_mode():
        for i in tqdm(range(generation_start_index, len(model_inputs_sorted)), desc=f"Generating generations ({n_generations_per_sample} generations per sample)"):
            for _ in range(n_generations_per_sample):
                model_outputs_text_all[idx_sorted[i]].append(\
                    watermark.generate_watermarked_text(model_inputs_sorted[i], **config['watermark']['generation_kwargs']) if config['watermark']['active'] \
                        else watermark.generate_unwatermarked_text(model_inputs_sorted[i], **config['watermark']['generation_kwargs'])
                )

            if i % config['generation'].get('save_every', 1) == 0 or \
                    i == len(model_inputs_sorted)-1:
                save_results_as_jsonl(
                    logger,
                    config,
                    {
                        "filled_prompt": filled_prompts, 
                        "watermarked_model_output": model_outputs_text_all, 
                        "config": config_dumped,
                        "server": current_server,
                        "vocab_size": vocab_size_dumped,
                        },
                    sample_df,
                    )

    # Finish
    logger.info(f"Generation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)