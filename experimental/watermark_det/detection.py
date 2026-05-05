from common.utils import get_logger, load_config, save_results_as_jsonl, Preprocessor

logger = get_logger(__name__)

import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

REQUIRED_CONFIG_FIELDS = [
    "model/name",
    "watermark",
    "file/root_dir",
    "file/generation/path",
    "file/generation/field",
    "file/output/path",
]


def main(args):
    # Load config
    config = load_config(logger, args.config, REQUIRED_CONFIG_FIELDS)

    # Load generations
    generation_config = config['file']['generation']
    logger.info(f"Loading generations from {generation_config['path']}")
    generation_df = pd.read_json(generation_config['path'], orient="records", lines=True)
    generation_df = generation_df.head(config['watermark'].get('max_samples', None))
    preprocess = generation_config.get('preprocess', [])
    preprocessor = Preprocessor(generation_df, generation_config['field'])
    for proc in preprocess:
        logger.info(f"Preprocessing generations using {proc['name']}")
        preprocessor(proc['name'], **proc['kwargs'])
    detection_targets:list[str] = preprocessor.get_preprocessed_generations(**generation_config.get('extract', {}))

    # Server (device) check
    if 'server' not in generation_df.columns:
        logger.warning(f"⚠️ Generation server is not found in the generations, so skipping server check. This may cause unexpected behavior if the server is not the same as the detection server.")
    else:
        generation_server_cands = generation_df['server'].unique()
        if len(generation_server_cands) > 1:
            raise ValueError(f"Generation servers are not consistent: {generation_server_cands}.")
        generation_server = generation_server_cands[0]
        detection_server = os.environ.get('SERVER_NAME', 'Unknown')
        if generation_server != detection_server:
            raise ValueError(f"Generation server {generation_server} is not the same as detection server {detection_server}. Please ensure to use the same server for generation and detection.")
        if generation_server == 'Unknown':
            raise ValueError(f"Generation server is unknown. Please confirm the server you used to generate the generations.")

    # Setup model
    # ref: https://github.com/redwyd/SymMark/blob/f68a52ae4ad34632222aadcc96770af84b4c51ae/main.py#L74-L86
    model_name = config['model']['name']
    logger.info(f"Setting up model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu"
    
    vocab_size = config['model'].get('vocab_size', None)
    if vocab_size is None:
        if 'vocab_size' in generation_df.columns:
            vocab_size_cands = generation_df['vocab_size'].apply(lambda vs_conf: eval(vs_conf)['size']).unique()
            if len(vocab_size_cands) != 1:
                raise ValueError(f"Vocab sizes are not consistent: {vocab_size_cands}.")
            vocab_size = int(vocab_size_cands[0])
        else:
            raise ValueError(f"Vocab size is not found in the model configuration and the generations. Please specify the vocab size in the model configuration.")
    
    transformers_config = TransformersConfig(
        model=None,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        **config['model']['transformers_config_kawrgs'],
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

    # Run detection
    logger.info(f"Running detection")
    detection_results=[]
    for generation in tqdm(detection_targets, mininterval=2.0):
        try:
            det_res = watermark.detect_watermark(generation)
            det_res["error"] = None
        except Exception as e:
            logger.error(f"Error detecting watermark for generation: {generation}, so skipping.")
            det_res = {"error": str(e)}
        detection_results.append(det_res)
    detection_results_df = pd.DataFrame(detection_results)

    # Save results
    save_results_as_jsonl(
        logger,
        config,
        {
            "detection_targets": detection_targets,
            "config": json.dumps(config),
            "server": os.environ.get('SERVER_NAME', 'Unknown'),
            }, 
        detection_results_df
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)