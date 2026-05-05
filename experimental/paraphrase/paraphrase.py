from common.utils import get_logger, load_config, save_results_as_jsonl, Preprocessor

logger = get_logger(__name__)

import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm
import pandas as pd
import time

from dipper_paraphrases.paraphrase_minimal import DipperParaphraser


REQUIRED_CONFIG_FIELDS = [

]


def main(args):
    # load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)

    # load generations
    generation_config = config['file']['generation']
    logger.info(f"Loading generations from {generation_config['path']}")
    generation_df = pd.read_json(generation_config['path'], orient="records", lines=True).head(config['paraphrase'].get('max_samples', None))
    preprocess = config['paraphrase'].get('preprocess', [])
    preprocessor = Preprocessor(generation_df, generation_config['field'])
    for proc in preprocess:
        logger.info(f"Preprocessing generations using {proc['name']}")
        preprocessor(proc['name'], **proc['kwargs'])
    paraphrase_targets = preprocessor.get_preprocessed_generations(**config['paraphrase'].get('extract', {}))

    # paraphrase generations
    logger.info(f"Paraphrasing generations")
    dipper_paraphraser = DipperParaphraser(verbose=False, model_kwargs=config['paraphrase']['model_kwargs'])
    paraphrase_func = lambda x: dipper_paraphraser.paraphrase(x, **config['paraphrase']['generation_kwargs'])
    paraphrase_outputs = []
    for tgts in tqdm(paraphrase_targets, desc="Paraphrasing generations"):
        is_list = isinstance(tgts, list)
        tgts_lst = tgts if is_list else [tgts]
        outs = [paraphrase_func(t) for t in tgts_lst]
        paraphrase_outputs.append(outs if is_list else outs[0])

    # save paraphrased generations
    logger.info(f"Saving paraphrased generations")
    save_results_as_jsonl(
        logger,
        config,
        {
            "paraphrase_outputs": paraphrase_outputs,
            "paraphrase_config": json.dumps(config)
        },
        generation_df,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)