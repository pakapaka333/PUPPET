"""
Prepare the test dataset for evaluation.

Reuses build_dataset() from 2_build_dataset/1_get_base_dataset.py so that
the same dataset construction logic is applied to both train and test splits.
The test set uses a different index range (start_index) to avoid overlap with
the training set.

To use your own dataset, modify build_dataset() in 1_get_base_dataset.py.
No changes are needed here.
"""
import argparse
import importlib.util
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

# Import build_dataset from 2_build_dataset/1_get_base_dataset.py.
# To switch datasets, modify build_dataset() in that file — no changes needed here.
_get_ds_path = SCRIPTS_ROOT / "2_build_dataset" / "1_get_base_dataset.py"
_spec = importlib.util.spec_from_file_location("get_base_dataset", _get_ds_path)
_get_ds_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_get_ds_mod)
build_dataset = _get_ds_mod.build_dataset


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = SCRIPTS_ROOT / config["output_dir"]

    ds = build_dataset(
        seed=config.get("seed", 42),
        num_samples=config.get("num_samples", 500),
        start_index=config.get("start_index", 5000),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving test dataset to {output_dir} ...")
    ds.save_to_disk(str(output_dir))
    logger.info(f"Done. {len(ds):,} test samples saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
