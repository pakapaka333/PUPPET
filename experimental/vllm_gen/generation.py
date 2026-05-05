from common.utils import get_logger, load_config, save_results_as_jsonl, BenchmarkManager

logger = get_logger(__name__)

import argparse
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

REQUIRED_CONFIG_FIELDS = [
    "model/name",
    "file/root_dir",
    "file/input/benchmark",
    "file/input/dataset",
    "file/output/path",
]


def __truncate_prompts(prompts_or_conversations:list[str] | list[list[dict]], is_inst_model:bool, max_new_tokens:int, max_model_length:int, tokenizer:AutoTokenizer) -> list[str] | list[list[dict]]:
    if is_inst_model:
        empty_conversation = [{"role": "user", "content": ""}]
        tokenized_empty_conversation = tokenizer.apply_chat_template(empty_conversation, return_tensors="pt", return_dict=True, add_special_tokens=True)
        template_length = tokenized_empty_conversation.input_ids.shape[1]
    else:
        template_length = 0
    
    max_prompt_length = max_model_length - max_new_tokens - template_length
    if max_prompt_length <= 0:
        raise ValueError(f"max_new_tokens ({max_new_tokens}) + template_length ({template_length}) must be smaller than max_model_length ({max_model_length})")

    half_max_prompt_length = int(max_prompt_length/2)
    head_length = half_max_prompt_length
    tail_length = max_prompt_length - half_max_prompt_length

    truncated_prompts_or_conversations = []; truncate_count = 0
    for raw_input in prompts_or_conversations:
        if is_inst_model:
            if (not isinstance(raw_input, list)) or (len(raw_input) != 1) or (raw_input[0].get('role', None) != 'user'):
                raise ValueError(f"Invalid input for instruct models: {raw_input}")
            tokenized_input = tokenizer.apply_chat_template(raw_input, return_tensors="pt", return_dict=True, add_special_tokens=True)
            if tokenized_input.input_ids.shape[1] > max_prompt_length:
                tokenized_content = tokenizer(raw_input[0]['content'], return_tensors="pt", add_special_tokens=True)['input_ids']
                first_half = tokenizer.decode(tokenized_content[0][:head_length], skip_special_tokens=True)
                second_half = tokenizer.decode(tokenized_content[0][-tail_length:], skip_special_tokens=True)
                truncated_content = first_half + second_half
                truncated_prompts_or_conversations.append([{"role": "user", "content": truncated_content}])
                truncate_count += 1 
            else:
                truncated_prompts_or_conversations.append(raw_input)
        else:
            if not isinstance(raw_input, str):
                raise ValueError(f"Invalid input for base models: {raw_input}")
            tokenized_input = tokenizer(raw_input, return_tensors="pt", add_special_tokens=True)
            if tokenized_input.input_ids.shape[1] > max_prompt_length:
                first_half = tokenizer.decode(tokenized_input.input_ids[0][:head_length], skip_special_tokens=True)
                second_half = tokenizer.decode(tokenized_input.input_ids[0][-tail_length:], skip_special_tokens=True)
                truncated_content = first_half + second_half
                truncated_prompts_or_conversations.append(truncated_content)
                truncate_count += 1
            else:
                truncated_prompts_or_conversations.append(raw_input)

    if truncate_count > 0:
        logger.warning(f"Truncated {truncate_count} prompts")
    else:
        logger.info(f"No prompts needed truncation")

    return truncated_prompts_or_conversations


def main(args):
    # Load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)

    # Load samples
    bm = BenchmarkManager()
    sample_df, conversations, filled_prompts = bm.get_samples_and_conversations(logger, config, return_filled_prompts=True)
    min_new_tokens, max_new_tokens = bm.get_min_max_lengths(config['file']['input']['dataset'])


    # Setup model
    use_lora = config['model'].get('lora_path', None) is not None
    model_name = config['model']['name']
    logger.info(f"Setting up model {model_name}")
    model = LLM(model=model_name, enable_lora=use_lora, **config['model']['kwargs'])
    merged_sampling_params = {
        "min_tokens": min_new_tokens,
        "max_tokens": max_new_tokens,
        **config['generation']['kwargs']
    }
    sampling_params = SamplingParams(**merged_sampling_params)
    tokenizer = model.get_tokenizer()
    is_inst_model = tokenizer.chat_template is not None
    max_model_length = getattr(model.llm_engine.engine_core.vllm_config.model_config, "max_model_len", None)
    if max_model_length is None:
        raise ValueError(f"Failed to find max model length in the model engine")


    # Truncate prompts
    use_chat_template = config['generation'].get('use_chat_template', is_inst_model)
    if use_chat_template:
        conversations = __truncate_prompts(conversations, use_chat_template, max_new_tokens, max_model_length, tokenizer)
    else:
        filled_prompts = __truncate_prompts(filled_prompts, use_chat_template, max_new_tokens, max_model_length, tokenizer)


    # Run generations
    logger.info(f"Running generations: {len(conversations)} samples, {config['generation']['kwargs']['n']} generations per sample.")
    chat_template_kwargs = config['generation'].get('chat_template_kwargs', {})
    if use_lora:
        lora_request = LoRARequest("dpo", 1, config['model']['lora_path'])
        model_outputs = model.chat(conversations, sampling_params, lora_request=lora_request, chat_template_kwargs=chat_template_kwargs)\
            if use_chat_template \
            else model.generate(filled_prompts, sampling_params, lora_request=lora_request)
    else:
        model_outputs = model.chat(conversations, sampling_params, chat_template_kwargs=chat_template_kwargs) \
            if use_chat_template \
            else model.generate(filled_prompts, sampling_params)
    model_outputs_text = [ [cand.text for cand in out.outputs] for out in model_outputs ]

    # Finish
    logger.info(f"Generation finished")
    save_results_as_jsonl(
        logger,
        config, 
        {
            "filled_prompt": filled_prompts,
            "vllm_model_output": model_outputs_text, 
            "config": json.dumps(config)
            },
        sample_df,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)