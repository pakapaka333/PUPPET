from common.utils import get_logger, load_config, Preprocessor, get_output_path

logger = get_logger(__name__)

import argparse
import math
import numpy as np
import joblib
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

from vllm import LLM, SamplingParams
import shap


class OpenAIDetectorSHAP:
    def __init__(self, **kwargs):
        # Initialize detector
        self.detector_name = "openai-community/roberta-large-openai-detector"
        self.detector_max_len = 512
        self.model = AutoModelForSequenceClassification.from_pretrained(self.detector_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.detector_name)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            return_all_scores=True,
            truncation=True,
            padding=True,
            max_length=self.detector_max_len,
        )
        self.shap_output_names = ["Machine", "Human"]

    def shap(self, shap_targets:list[str]|list[list[str]], shap_reference:list[str]):
        explainer = shap.Explainer(self.pipeline)
        if isinstance(shap_targets[0], list):
            shap_results = [explainer(shap_tgt, output_names=self.shap_output_names) for shap_tgt in shap_targets]
        elif isinstance(shap_targets[0], str):
            shap_results = explainer(shap_targets, output_names=self.shap_output_names)
        return shap_results


class LLMAsAJudgeSHAP:
    def __init__(self, **kwargs):
        # Initialize judge model
        judge_model_config = kwargs.get("model")
        if (not judge_model_config) or (not judge_model_config.get("name")):
            raise ValueError("judge_model is required.")
        self.judge_model = LLM(
            model=judge_model_config['name'],
            **judge_model_config.get("kwargs", {}),
        )
        self.judge_tokenizer = self.judge_model.get_tokenizer()

        # Initialize thinking parameters
        thinking_kwargs = kwargs.get("thinking_kwargs", {})
        if not thinking_kwargs:
            raise ValueError("thinking_kwargs is required.")
        
        thinking_kwargs.setdefault("include_stop_str_in_output", True)
        thinking_kwargs.setdefault("skip_special_tokens", False)

        if thinking_kwargs["include_stop_str_in_output"] is not True:
            raise ValueError("include_stop_str_in_output must be True for logprobs scoring.")
        if thinking_kwargs["skip_special_tokens"] is not False:
            raise ValueError("skip_special_tokens must be False for logprobs scoring.")
        
        self.thinking_params = SamplingParams(**thinking_kwargs)

        # Initialize prompt
        self.prompt_src = kwargs.get("prompt", None)
        if self.prompt_src is None:
            raise ValueError("prompt is required.")
        if os.path.isfile(self.prompt_src):
            with open(self.prompt_src, "r", encoding="utf-8") as f:
                self.prompt = f.read()
        else:
            self.prompt = self.prompt_src
        self.filled_prompt = ""
        self.chat_template_kwargs = kwargs.get("chat_template_kwargs", {})
        self.stop_seqs = thinking_kwargs.get("stop")
        if not self.stop_seqs:
            raise ValueError("stop tokens are required for logprobs retry (non-empty).")
        if isinstance(self.stop_seqs, str):
            self.stop_seqs = [self.stop_seqs]
        if (not isinstance(self.stop_seqs, list)) or (not all(isinstance(s, str) and s for s in self.stop_seqs)):
            raise ValueError("stop must be a non-empty str or list[str].")

        # Scoring parameters
        self.max_retries = kwargs.get("max_retries", 10)
        self.scoring_prefix = kwargs.get("scoring_prefix", "")
        if not self.scoring_prefix or not isinstance(self.scoring_prefix, str):
            raise ValueError("scoring_prefix must be a non-empty string.")
        
        self.scoring_batch_size = kwargs.get("scoring_batch_size", 1000)
        if self.scoring_batch_size <= 0:
            raise ValueError("scoring_batch_size must be a positive integer.")

        self.score_labels = kwargs.get("score_labels", [])
        if not self.score_labels or not isinstance(self.score_labels, list) or not all(isinstance(lbl, str) for lbl in self.score_labels):
            raise ValueError("score_labels must be a non-empty list of strings.")
        self.score_ints = [int(lbl) for lbl in self.score_labels]
        self.score_label_ids = [self.judge_tokenizer.encode(lbl, add_special_tokens=False) for lbl in self.score_labels]

        scoring_kwargs = dict(kwargs.get("scoring_kwargs", {}))
        if not scoring_kwargs:
            raise ValueError("scoring_kwargs is required.")

        scoring_kwargs.setdefault("max_tokens", 1)
        scoring_kwargs.setdefault("prompt_logprobs", 1)
        if scoring_kwargs["max_tokens"] != 1:
            raise ValueError("max_tokens must be 1 for logprobs scoring.")
        if scoring_kwargs["prompt_logprobs"] != 1:
            raise ValueError("prompt_logprobs must be 1 for logprobs scoring.")

        self.scoring_params = SamplingParams(**scoring_kwargs)
        self.default_score = kwargs.get("default_score", -1)

        self.shap_batch_size = kwargs.get("shap_batch_size", 1000)
        if self.shap_batch_size <= 0:
            raise ValueError("shap_batch_size must be a positive integer.")


    def _normalize_predictions(self, predictions: list[str]|list[list[str]]) -> tuple[list[list[str]], int]:
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


    def _llm_as_a_judge(self, predictions: list[str]|list[list[str]], contexts: list[str]) -> list[float]|list[list[float]]:
        # validate contexts
        if not isinstance(contexts, list) or not contexts or not isinstance(contexts[0], str):
            raise ValueError("contexts must be a non-empty list[str].")

        preds_2d, k = self._normalize_predictions(predictions)

        n = len(preds_2d)
        if n != len(contexts):
            raise ValueError(f"predictions and contexts must have the same length, but got {n} and {len(contexts)}.")

        # flatten (context, prediction) pairs
        flat_contexts: list[str] = []
        flat_predictions: list[str] = []
        for c, sub_preds in zip(contexts, preds_2d):
            flat_contexts.extend([c] * k)
            flat_predictions.extend(sub_preds)

        filled_prompts = [
            self.prompt.format(context=c, prediction=p)
            for c, p in tqdm(
                zip(flat_contexts, flat_predictions),
                mininterval=2.0,
                total=len(flat_contexts),
                desc="Filling prompts",
            )
        ]
        formatted_prompts = [[{"role": "user", "content": fp}] for fp in filled_prompts]

        # get thinking trajectories (+ retry if stop sequence is missing)
        m = len(formatted_prompts)  # = n * k
        judge_output_texts: list[str] = ["" for _ in range(m)]
        remaining = list(range(m))

        def _has_any_stop(text: str) -> bool:
            t = text.rstrip()
            return any(t.endswith(s) or t.endswith(s.rstrip()) for s in self.stop_seqs)

        # output judge_outputs as temporary .jsonl
        for retry_idx in range(self.max_retries):
            sub_prompts = [formatted_prompts[i] for i in remaining]
            judge_outputs = self.judge_model.chat(
                sub_prompts, self.thinking_params, chat_template_kwargs=self.chat_template_kwargs
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
                f"[{retry_idx+1}/{self.max_retries}]"
            )

        failed = set(remaining)
        if failed:
            logger.warning(
                f"Thinking generation failed for {len(failed)} samples "
                f"after {self.max_retries} retries. "
                f"Assigning score = -1 to them."
            )

        # prepare scoring prompts
        scoring_prefixes: list[str] = []
        valid_indices: list[int] = []
        for i, (prompt_msg, think) in enumerate(zip(formatted_prompts, judge_output_texts)):
                if i in failed:
                    continue
                pref = self.judge_tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=False) \
                        + think + self.scoring_prefix
                scoring_prefixes.append(pref)
                valid_indices.append(i)

        # compute scores
        def _softmax(logps: list[float]) -> list[float]:
            m = max(logps)
            exps = [math.exp(lp - m) for lp in logps]
            s = sum(exps)
            return [e / s for e in exps]

        N_valid = len(scoring_prefixes)
        L = len(self.score_labels)
        logps_2d: list[list[float]] = [[0.0] * L for _ in range(N_valid)]

        total_jobs = N_valid * L
        batch_prompts: list[str] = []
        batch_map: list[tuple[int, int]] = []

        def _flush_batch():
            nonlocal batch_prompts, batch_map
            if not batch_prompts:
                return
            outs = self.judge_model.generate(batch_prompts, self.scoring_params)
            if len(outs) != len(batch_prompts):
                raise RuntimeError(f"vLLM returned {len(outs)} outputs for {len(batch_prompts)} prompts.")
            for out, (i, j) in zip(outs, batch_map):
                plp = out.prompt_logprobs
                ptids = out.prompt_token_ids
                n_ids = len(self.score_label_ids[j])
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
                    batch_prompts.append(pref + self.score_labels[j])
                    batch_map.append((i, j))
                    if len(batch_prompts) >= self.scoring_batch_size:
                        _flush_batch()
                        pbar.update(self.scoring_batch_size)
            # flush remainder
            rem = len(batch_prompts)
            if rem:
                _flush_batch()
                pbar.update(rem)

        all_scores_flat = [self.default_score for _ in range(m)]
        for local_i, orig_i in enumerate(valid_indices):
            probs = _softmax(logps_2d[local_i])
            expected = sum(self.score_ints[j] * probs[j] for j in range(L))
            all_scores_flat[orig_i] = expected

        # reshape back
        if k == 1:
            return [all_scores_flat[i] for i in range(0, m, 1)]

        all_scores_2d: list[list[float]] = []
        for i in range(n):
            start = i * k
            all_scores_2d.append(all_scores_flat[start:start + k])

        return all_scores_2d
    

    def shap(self, shap_targets:list[str]|list[list[str]], shap_reference:list[str]):
        if not isinstance(shap_targets, list) or len(shap_targets) == 0:
            raise ValueError("shap_targets must be a non-empty list.")

        # Helper: build an explainer for a *fixed* context.
        def _build_explainer_for_context(context: str | None):
            ctx = context or ""

            def _model_fn(texts):
                if isinstance(texts, str):
                    texts_list = [texts]
                elif isinstance(texts, np.ndarray):
                    texts_list = [str(t) for t in texts.ravel().tolist()]
                else:
                    raise ValueError("texts must be a string, numpy.ndarray, or list.")

                texts_list = [str(t) for t in texts_list]
                if len(texts_list) == 0:
                    return np.zeros((0, 1), dtype=float)

                contexts = [ctx] * len(texts_list)
                scores = self._llm_as_a_judge(texts_list, contexts)

                if isinstance(scores, list) and scores and isinstance(scores[0], list):
                    scores = [s[0] for s in scores]

                return np.array(scores, dtype=float).reshape(-1, 1)

            text_masker = shap.maskers.Text(self.judge_tokenizer)
            return shap.Explainer(_model_fn, text_masker)

        # Case 1: list[list[str]]
        if isinstance(shap_targets[0], list):
            if len(shap_reference) == 1:
                # Single shared context.
                shap_results = []
                explainer = _build_explainer_for_context(shap_reference[0])
                for preds in shap_targets:
                    shap_results.append(explainer(preds, batch_size=self.shap_batch_size))
                return shap_results

            if len(shap_reference) != len(shap_targets):
                raise ValueError(
                    "When shap_targets is list[list[str]], shap_reference must "
                    "have length 1 or len(shap_targets)."
                )

            # Per-example context
            shap_results = []
            for preds, ctx in zip(shap_targets, shap_reference):
                explainer = _build_explainer_for_context(ctx)
                shap_results.append(explainer(preds, batch_size=self.shap_batch_size))
            return shap_results

        # Case 2: list[str]
        if isinstance(shap_targets[0], str):
            # Shared context for all examples.
            if len(shap_reference) == 0:
                explainer = _build_explainer_for_context(None)
                return explainer(shap_targets, batch_size=self.shap_batch_size)

            if len(shap_reference) == 1:
                explainer = _build_explainer_for_context(shap_reference[0])
                return explainer(shap_targets, batch_size=self.shap_batch_size)

            # Per-example context -> return list[Explanation]
            if len(shap_reference) != len(shap_targets):
                raise ValueError(
                    "When shap_targets is list[str], shap_reference must "
                    "have length 0, 1, or len(shap_targets)."
                )

            shap_results = []
            for pred, ctx in zip(shap_targets, shap_reference):
                explainer = _build_explainer_for_context(ctx)
                shap_results.append(explainer([pred]))
            return shap_results

        raise ValueError("shap_targets must be list[str] or list[list[str]].")


NAME2MODEL_FN = {
    "openai": OpenAIDetectorSHAP,
    "llm_as_a_judge": LLMAsAJudgeSHAP,
}


def main(args):
    # Load config
    config = load_config(logger, args.config, [])

    # Load generations
    generation_config = config['file']['generation']
    logger.info(f"Loading generations from {generation_config['path']}")
    generation_df = pd.read_json(generation_config['path'], orient="records", lines=True).head(config['shap'].get('max_samples', None))
    preprocess = config['shap'].get('preprocess', {})
    preprocessor = Preprocessor(generation_df, generation_config['field'])
    for proc in preprocess:
        logger.info(f"Preprocessing generations using {proc['name']}")
        preprocessor(proc['name'], **proc['kwargs'])
    shap_targets = preprocessor.get_preprocessed_generations(**config['shap'].get('extract', {}))

    # Analyze
    logger.info(f"Analyzing metrics")
    shap_models = config['shap']['models']
    for model_name, model_config in shap_models.items():
        if model_config.get("reference"):
            reference_config = model_config['reference']
            logger.info(f"Loading references from {reference_config['path']}")
            reference_df = pd.read_json(reference_config['path'], orient="records", lines=True).head(config['shap'].get('max_samples', None))
            shap_reference   = reference_df[reference_config['field']].tolist()
        else:
            shap_reference = [None] * len(shap_targets)

        shap_analyzer = NAME2MODEL_FN[model_name](**model_config.get('kwargs', {}))
        shap_results = shap_analyzer.shap(shap_targets, shap_reference)

        # Save results
        joblib.dump(
            shap_results,
            get_output_path(config, extension=".joblib", custom_postfix=f"_{model_name}"),
            compress=3,
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)