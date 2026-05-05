from common.utils import get_logger, load_config, save_results_as_jsonl, Preprocessor

logger = get_logger(__name__)

import argparse
from argparse import Namespace
from datetime import datetime
import json
import os
import torch
from tqdm import tqdm
import pandas as pd
import time
from typing import Callable

ERROR_SCORE = -1.0

def rouge_score_waterbench(predictions:list[str]|list[list[str]], answers:list[list[str]], **kwargs) -> list[float]|list[list[float]]:
    """
    Calculate the best Rouge-L score for each prediction and answer pair. \\
    This implementation is based on the following WaterBench evaluation scripts:
    - WaterBench/eval.py
    - WaterBench/metrics.py
    """
    from rouge import Rouge

    rouge = Rouge()
    def __safe_get_rouge_score(prediction:str, ground_truth:str) -> float:
        try:
            rouge_scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        except:
            return 0.0
        return rouge_scores["rouge-l"]["f"]
    
    all_scores = []
    is_list = isinstance(predictions[0], list)
    for (pred_or_pred_list, ground_truths) in tqdm(zip(predictions, answers), mininterval=2.0, total=len(predictions), desc="Calculating Rouge-L scores"):
        pred_list = pred_or_pred_list if is_list else [pred_or_pred_list]
        scores = []
        for pred in pred_list:
            s = 0.
            for gt in ground_truths:
                s = max(s, __safe_get_rouge_score(pred, gt))
            scores.append(s)
        all_scores.append(scores if is_list else scores[0])
    return all_scores


def detect_hf_detector(
    model_name: str,
    detector_max_len: int,
    machine_class_idx: int,
    predictions:list[str]|list[list[str]],
    text_preprocessor: Callable[[str], str]|None = None,
    **kwargs
    ) -> list[float]|list[list[float]]:
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    detector_tokenizer = AutoTokenizer.from_pretrained(model_name)
    detector = AutoModelForSequenceClassification.from_pretrained(model_name, token=os.getenv("HF_TOKEN"), **kwargs.get("detector_kwargs", {}))

    is_list = isinstance(predictions[0], list)    

    if text_preprocessor is not None:
        logger.info("Found text preprocessor. Applying it to predictions.")
        preprocessed_predictions = []
        for pred_or_pred_list in tqdm(predictions, mininterval=2.0, total=len(predictions), desc=f"Applying text preprocessor, {text_preprocessor.__name__}"):
            pred_list = pred_or_pred_list if is_list else [pred_or_pred_list]
            preprocessed_pred_list = [text_preprocessor(pred) for pred in pred_list]
            preprocessed_predictions.append(preprocessed_pred_list if is_list else preprocessed_pred_list[0])
        predictions = preprocessed_predictions

    all_scores = []
    with torch.no_grad():
        for pred_or_pred_list in tqdm(predictions, mininterval=2.0, total=len(predictions), desc=f"Detecting by {model_name}"):
            pred_list = pred_or_pred_list if is_list else [pred_or_pred_list]
            enc = detector_tokenizer(pred_list, return_tensors='pt', truncation=True, padding=True, max_length=detector_max_len)
            enc = {k: v.to(detector.device) for k, v in enc.items()}
            logits = detector(**enc).logits
            machine_probs = F.softmax(logits, dim=-1)[:, machine_class_idx].detach().cpu().tolist()
            all_scores.append(machine_probs if is_list else machine_probs[0])

    return all_scores


def detect_openai(predictions:list[str]|list[list[str]], answers:list[list[str]], **kwargs) -> list[float]|list[list[float]]:
    detector_name = "openai-community/roberta-large-openai-detector"
    detector_max_len = 512  # ref: `"model_max_length": 512` in tokenizer_config.json. `"max_position_embeddings": 514` in model_config.json.
    machine_class_idx = 0 # `"id2label": { "0": "Fake", "1": "Real"}` in config.json.
    return detect_hf_detector(detector_name, detector_max_len, machine_class_idx, predictions, **kwargs)


def detect_fakespotai(predictions:list[str]|list[list[str]], answers:list[list[str]], **kwargs) -> list[float]|list[list[float]]:
    from materials.utils import preprocess_fakepostai
    detector_name = "fakespot-ai/roberta-base-ai-text-detection-v1"
    detector_max_len = 512  # ref: `"model_max_length": 512` in tokenizer_config.json. `"max_position_embeddings": 514` in config.json.
    machine_class_idx = 1   # `"id2label": { "0": "Human", "1": "AI"}` in config.json.
    return detect_hf_detector(detector_name, detector_max_len, machine_class_idx, predictions, text_preprocessor=preprocess_fakepostai, **kwargs)


def detect_mage(predictions:list[str]|list[list[str]], answers:list[list[str]], **kwargs) -> list[float]|list[list[float]]:
    from materials.utils import preprocess_mage
    detector_name = "yaful/MAGE"
    detector_max_len = 512  # to match the other detectors
    machine_class_idx = 0   # `"label2decisions": { 0: "machine-generated", 1: "human-written"}` in detect in utils.py.
    return detect_hf_detector(detector_name, detector_max_len, machine_class_idx, predictions, text_preprocessor=preprocess_mage, **kwargs)


def _normalize_predictions(predictions: list[str]|list[list[str]]) -> tuple[list[list[str]], int]:
    """
    Returns:
      preds_2d: shape (n, k)
      k: number of predictions per sample
    """
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions must be a non-empty list.")

    # predictions is a list of strings
    if isinstance(predictions[0], str):
        preds_2d = [[p] for p in predictions]
        return preds_2d, 1

    # predictions is a list of lists of strings
    if isinstance(predictions[0], list):
        preds_2d = predictions  # type: ignore[assignment]
        if not preds_2d or not isinstance(preds_2d[0], list) or not preds_2d[0]:
            raise ValueError("predictions must be list[str] or list[list[str]] with non-empty sublists.")

        # check that all sublists contain only strings
        if not isinstance(preds_2d[0][0], str):
            raise ValueError("predictions must contain strings.")

        k = len(preds_2d[0])
        if k == 0:
            raise ValueError("Each sublist in predictions must be non-empty.")
        
        # check that all sublists have the same length
        for i, sub in enumerate(preds_2d):
            if not isinstance(sub, list):
                raise ValueError(f"predictions[{i}] must be a list of strings.")
            if len(sub) != k:
                raise ValueError(f"All sublists must have the same length. predictions[0] has {k}, but predictions[{i}] has {len(sub)}.")
            if any(not isinstance(x, str) for x in sub):
                raise ValueError(f"predictions[{i}] contains non-string elements.")
        return preds_2d, k

    raise ValueError("predictions must be list[str] or list[list[str]].")


def llm_as_a_judge(predictions: list[str]|list[list[str]], contexts: list[str], **kwargs) -> list[float]|list[list[float]]:
    import math
    from vllm import LLM, SamplingParams
    
    # validate contexts
    if not isinstance(contexts, list) or not contexts or not isinstance(contexts[0], str):
        raise ValueError("contexts must be a non-empty list[str].")

    preds_2d, k = _normalize_predictions(predictions)

    n = len(preds_2d)
    if n != len(contexts):
        raise ValueError(f"predictions and contexts must have the same length, but got {n} and {len(contexts)}.")

    # prompt
    prompt_src = kwargs.get("prompt", None)
    if (not prompt_src) or (not isinstance(prompt_src, str)):
        raise ValueError("Prompt is required.")
    if os.path.isfile(prompt_src):
        with open(prompt_src, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = prompt_src

    # flatten (context, prediction) pairs
    flat_contexts: list[str] = []
    flat_predictions: list[str] = []
    for c, sub_preds in zip(contexts, preds_2d):
        flat_contexts.extend([c] * k)
        flat_predictions.extend(sub_preds)

    filled_prompts = [
        prompt.format(context=c, prediction=p)
        for c, p in tqdm(
            zip(flat_contexts, flat_predictions),
            mininterval=2.0,
            total=len(flat_contexts),
            desc="Filling prompts",
        )
    ]
    use_chat = kwargs.get("use_chat_template", False)
    if not use_chat:
        raise ValueError("logprobs scoring method requires use_chat_template to be True.")
    formatted_prompts = [[{"role": "user", "content": fp}] for fp in filled_prompts]

    # model
    judge_model_config = kwargs.get("model", None)
    if (not judge_model_config) or (not judge_model_config.get("name")):
        raise ValueError("Judge model's name is required.")
    judge_model = LLM(model=judge_model_config["name"], **judge_model_config.get("kwargs", {}))

    thinking_kwargs = dict(kwargs.get("thinking_kwargs", {}))
    if not thinking_kwargs:
        raise ValueError("Thinking config is required.")
    
    thinking_kwargs.setdefault("include_stop_str_in_output", True)
    thinking_kwargs.setdefault("skip_special_tokens", False)

    if thinking_kwargs["include_stop_str_in_output"] is not True:
        raise ValueError("include_stop_str_in_output must be True for logprobs scoring.")
    if thinking_kwargs["skip_special_tokens"] is not False:
        raise ValueError("skip_special_tokens must be False for logprobs scoring.")
    
    thinking_params = SamplingParams(**thinking_kwargs)

    # get thinking trajectories (+ retry if stop sequence is missing)
    m = len(formatted_prompts)  # = n * k
    judge_output_texts: list[str] = ["" for _ in range(m)]
    remaining = list(range(m))

    max_retries = kwargs.get("max_retries", 10)
    chat_template_kwargs = thinking_kwargs.get("chat_template_kwargs", {})

    stop_seqs = thinking_kwargs.get("stop", None)
    if not stop_seqs:
        raise ValueError("stop tokens are required for logprobs retry (non-empty).")
    if isinstance(stop_seqs, str):
        stop_seqs = [stop_seqs]
    if (not isinstance(stop_seqs, list)) or (not all(isinstance(s, str) and s for s in stop_seqs)):
        raise ValueError("stop must be a non-empty str or list[str].")

    def _has_any_stop(text: str) -> bool:
        t = text.rstrip()
        return any(t.endswith(s) or t.endswith(s.rstrip()) for s in stop_seqs)

    # output judge_outputs as temporary .jsonl
    judge_outputs_path = kwargs.get("judge_outputs_path", None)
    if (judge_outputs_path is not None) and (not isinstance(judge_outputs_path, str)):
        raise ValueError("judge_outputs_path must be a str path if provided.")

    def _read_judge_outputs_jsonl(path: str, m_expected: int) -> tuple[list[str], set[int]]:
        texts = [""] * m_expected
        failed_local: set[int] = set()

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line must be an object at {path}:{line_no}")

                if "index" not in obj:
                    raise ValueError(f"Missing 'index' field at {path}:{line_no}")
                idx = obj["index"]
                if not isinstance(idx, int):
                    raise ValueError(f"'index' must be int at {path}:{line_no}")
                if idx < 0 or idx >= m_expected:
                    raise ValueError(f"'index' out of range (0..{m_expected-1}) at {path}:{line_no}")

                txt = obj.get("text", "")
                if txt is None:
                    txt = ""
                if not isinstance(txt, str):
                    raise ValueError(f"'text' must be str at {path}:{line_no}")
                texts[idx] = txt

                failed_flag = obj.get("failed", False)
                if failed_flag is True:
                    failed_local.add(idx)

        return texts, failed_local

    def _write_judge_outputs_jsonl(path: str, texts: list[str], failed_set: set[int]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                rec = {"index": i, "text": t, "failed": (i in failed_set)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if (judge_outputs_path is not None) and os.path.isfile(judge_outputs_path):
        # load existing judge_outputs from .jsonl
        logger.info(f"Loading existing judge_outputs from: {judge_outputs_path}")
        judge_output_texts, failed = _read_judge_outputs_jsonl(judge_outputs_path, m)
    
    else:
        # generate new judge_outputs
        for retry_idx in range(max_retries):
            sub_prompts = [formatted_prompts[i] for i in remaining]
            judge_outputs = judge_model.chat(
                sub_prompts, thinking_params, chat_template_kwargs=chat_template_kwargs
            )

            new_remaining = []
            for local_i, out in enumerate(judge_outputs):
                orig_i = remaining[local_i]
                out_text = out.outputs[0].text

                if _has_any_stop(out_text):
                    judge_output_texts[orig_i] = out_text
                else:
                    new_remaining.append(orig_i)

            remaining = new_remaining
            if not remaining:
                break

            logger.info(
                f"Retrying thinking generation for {len(remaining)} samples "
                f"[{retry_idx+1}/{max_retries}]"
            )

        failed = set(remaining)
        if failed:
            logger.warning(
                f"Thinking generation failed for {len(failed)} samples "
                f"after {max_retries} retries. "
                f"Assigning score = -1 to them."
            )

        judge_outputs_path = os.path.join(
            kwargs.get("judge_outputs_dir", "."), 
            f"judge_outputs_{os.environ.get("RUN_NAME", "evaluation")}_{int(time.time())}.jsonl"
        )
        _write_judge_outputs_jsonl(judge_outputs_path, judge_output_texts, failed)
        logger.info(f"Saved temporary judge_outputs to: {judge_outputs_path}")

    # prepare scoring prompts
    score_prefix = kwargs.get("score_prefix", "")
    if not score_prefix or not isinstance(score_prefix, str):
        raise ValueError("score_prefix must be a non-empty string.")
    judge_tokenizer = judge_model.get_tokenizer()

    scoring_batch_size = int(kwargs.get("scoring_batch_size", 1000))
    if scoring_batch_size <= 0:
        raise ValueError("scoring_batch_size must be a positive integer.")

    scoring_prefixes: list[str] = []
    valid_indices: list[int] = []
    for i, (prompt_msg, think) in enumerate(zip(formatted_prompts, judge_output_texts)):
            if i in failed:
                continue
            pref = judge_tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=False) \
                     + think + score_prefix
            scoring_prefixes.append(pref)
            valid_indices.append(i)

    score_labels = kwargs.get("score_labels", [])
    if not score_labels or not isinstance(score_labels, list) or not all(isinstance(lbl, str) for lbl in score_labels):
        raise ValueError("score_labels must be a non-empty list of strings.")
    score_ints = [int(lbl) for lbl in score_labels]
    score_label_ids = [judge_tokenizer.encode(lbl, add_special_tokens=False) for lbl in score_labels]

    # compute logprobs
    scoring_kwargs = dict(kwargs.get("scoring_kwargs", {}))
    if not scoring_kwargs:
        raise ValueError("Scoring config is required.")

    scoring_kwargs.setdefault("max_tokens", 1)
    scoring_kwargs.setdefault("prompt_logprobs", 1)
    if scoring_kwargs["max_tokens"] != 1:
        raise ValueError("max_tokens must be 1 for logprobs scoring.")
    if scoring_kwargs["prompt_logprobs"] != 1:
        raise ValueError("prompt_logprobs must be 1 for logprobs scoring.")

    scoring_params = SamplingParams(**scoring_kwargs)

    # compute scores
    def _softmax(logps: list[float]) -> list[float]:
        m = max(logps)
        exps = [math.exp(lp - m) for lp in logps]
        s = sum(exps)
        return [e / s for e in exps]

    N_valid = len(scoring_prefixes)
    L = len(score_labels)
    logps_2d: list[list[float]] = [[0.0] * L for _ in range(N_valid)]

    total_jobs = N_valid * L
    batch_prompts: list[str] = []
    batch_map: list[tuple[int, int]] = []

    def _flush_batch():
        nonlocal batch_prompts, batch_map
        if not batch_prompts:
            return
        outs = judge_model.generate(batch_prompts, scoring_params)
        if len(outs) != len(batch_prompts):
            raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(batch_prompts)} prompts.")
        for out, (i, j) in zip(outs, batch_map):
            plp = out.prompt_logprobs
            ptids = out.prompt_token_ids
            n_ids = len(score_label_ids[j])
            start = len(ptids) - n_ids
            s = 0.0
            for pos in range(start, len(ptids)):
                d = plp[pos]
                if d is None:
                    raise RuntimeError(f"prompt_logprobs[{pos}] is None (pos={pos}).")
                tok_id = ptids[pos]
                lp_obj = d.get(tok_id)
                if lp_obj is None:
                    raise RuntimeError(
                        f"token_id={tok_id} not found in prompt_logprobs[{pos}]. Increase prompt_logprobs."
                    )
                s += lp_obj.logprob
            logps_2d[i][j] = s
        batch_prompts = []
        batch_map = []

    with tqdm(total=total_jobs, desc="Scoring (logprobs)", mininterval=2.0) as pbar:
        for i in range(N_valid):
            pref = scoring_prefixes[i]
            for j in range(L):
                batch_prompts.append(pref + score_labels[j])
                batch_map.append((i, j))
                if len(batch_prompts) >= scoring_batch_size:
                    _flush_batch()
                    pbar.update(scoring_batch_size)
        # flush remainder
        rem = len(batch_prompts)
        if rem:
            _flush_batch()
            pbar.update(rem)

    all_scores_flat = [ERROR_SCORE for _ in range(m)]
    for local_i, orig_i in enumerate(valid_indices):
        probs = _softmax(logps_2d[local_i])
        expected = sum(score_ints[j] * probs[j] for j in range(L))
        all_scores_flat[orig_i] = expected

    # reshape back
    if k == 1:
        return [all_scores_flat[i] for i in range(0, m, 1)]

    all_scores_2d: list[list[float]] = []
    for i in range(n):
        start = i * k
        all_scores_2d.append(all_scores_flat[start:start + k])

    return all_scores_2d


NAME2FUNCTION = {
    "rouge_score_waterbench": rouge_score_waterbench,
    "detect_openai": detect_openai,
    "detect_fakespotai": detect_fakespotai,
    "detect_mage": detect_mage,
    "llm_as_a_judge": llm_as_a_judge,
}


REQUIRED_CONFIG_FIELDS = [
    "evaluation/metrics",
    "file/output/path",
    "file/generation/path",
    "file/generation/field"
]


def main(args):
    # Load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)

    # Load generations
    generation_config = config['file']['generation']
    logger.info(f"Loading generations from {generation_config['path']}")
    generation_df = pd.read_json(generation_config['path'], orient="records", lines=True).head(config['evaluation'].get('max_samples', None))
    preprocess = config['evaluation'].get('preprocess', {})
    preprocessor = Preprocessor(generation_df, generation_config['field'])
    for proc in preprocess:
        logger.info(f"Preprocessing generations using {proc['name']}")
        preprocessor(proc['name'], **proc['kwargs'])
    evaluation_targets = preprocessor.get_preprocessed_generations(**config['evaluation'].get('extract', {}))

    # Evaluate
    logger.info(f"Evaluating metrics")
    evaluation_metrics = config['evaluation']['metrics']
    evaluation_results = {}; reference_answers = {}
    for mtr_name, mtr_config in evaluation_metrics.items():
        # Load references
        if mtr_config.get('reference', None) is not None:
            reference_config = mtr_config['reference']
            logger.info(f"Found reference file for metric {mtr_name}. Loading references from {reference_config['path']}")
            reference_df = pd.read_json(reference_config['path'], orient="records", lines=True).head(config['evaluation'].get('max_samples', None))
            reference_answers[f"reference_{mtr_name}"] = reference_df[reference_config['field']].tolist()
        else:
            reference_answers[f"reference_{mtr_name}"] = [None] * len(evaluation_targets)

        # Run evaluation
        evaluation_results[mtr_name] = NAME2FUNCTION[mtr_name](evaluation_targets, reference_answers[f"reference_{mtr_name}"], **mtr_config.get('kwargs', {}))

        if mtr_name == "llm_as_a_judge_request":
            logger.info("Terminated without saving results since llm_as_a_judge_request utilizes the batch API.")
            break

        # Save results
        save_results_as_jsonl(
            logger,
            config,
            {
                "evaluation_targets": evaluation_targets,
                "config": json.dumps(config),
                **evaluation_results,
                **reference_answers,
            } 
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)