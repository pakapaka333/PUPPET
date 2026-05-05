#!/usr/bin/env python3
"""Download and prepare external files required for evaluation/materials/.

Usage:
    bash experimental/setup/setup.sh
Or directly:
    python experimental/setup/scripts/setup_eval.py
"""

import json
import urllib.request
from pathlib import Path

MATERIALS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "materials"

NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/PON2020/IELTSWriting"
    "/61d284bbaa7b494cf5a43775d319d46ca1999ffe"
    "/study_2_code/gpt4o_without_example.ipynb"
)

UTILS_DOWNLOADS = [
    {
        "url": "https://raw.githubusercontent.com/yafuly/MAGE/main/deployment/utils.py",
        "dest": MATERIALS_DIR / "utils" / "mage_utils.py",
    },
    {
        "url": "https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/resolve/main/utils.py",
        "dest": MATERIALS_DIR / "utils" / "fakepostai_utils.py",
    },
]


def build_ielts_prompt():
    """Download the IELTSWriting notebook and extract + edit the prompt template.

    Edits applied to the original prompt_template:
      1. {prompt_text} -> {context}
      2. {essay_text}  -> {prediction}
      3. Remove " with justification." from the expected response format line
    """
    dest = MATERIALS_DIR / "llm_as_a_judge_prompts" / "ielts.txt"

    print(f"Downloading {NOTEBOOK_URL}")
    with urllib.request.urlopen(NOTEBOOK_URL) as response:
        notebook = json.loads(response.read().decode("utf-8"))

    # Find the cell containing prompt_template
    prompt_template = None
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "prompt_template" in source and "Band descriptors" in source:
            # Extract the string value of prompt_template
            # It is assigned as:  prompt_template = """..."""
            start = source.index('"""', source.index("prompt_template")) + 3
            end = source.index('"""', start)
            prompt_template = source[start:end]
            break

    if prompt_template is None:
        raise RuntimeError("Could not find prompt_template in notebook")

    # Apply edits
    prompt_template = prompt_template.replace("{prompt_text}", "{context}")
    prompt_template = prompt_template.replace("{essay_text}", "{prediction}")
    prompt_template = prompt_template.replace(" with justification.", "")

    # Strip leading/trailing newlines added by the triple-quote definition
    prompt_template = prompt_template.strip("\n")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(prompt_template, encoding="utf-8")
    print(f"  -> {dest}")


def download_utils():
    for entry in UTILS_DOWNLOADS:
        url = entry["url"]
        dest: Path = entry["dest"]

        print(f"Downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")
        dest.write_text(content, encoding="utf-8")
        print(f"  -> {dest}")


def main():
    build_ielts_prompt()
    download_utils()
    print("Done.")


if __name__ == "__main__":
    main()
