from datasets import load_from_disk
import logging
import os
import pandas as pd
import re
import yaml

def get_logger(name:str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(logger:logging.Logger, config_path:str, required_fields:list[str]) -> dict:
    # load
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # check
    for field in required_fields:
        c = config.copy()
        for f in field.split("/"):
            c = c.get(f, None)
            if c is None:
                raise ValueError(f"Required field {field} is not set in the config file. Current config: {c}")
    return config


class BenchmarkManager():
    ROOT_DIR = "PUPPET/experimental"
    WATERBENCH_CONFIG = {
        # The min_length is based on https://github.com/redwyd/SymMark/blob/f68a52ae4ad34632222aadcc96770af84b4c51ae/config/DT.json.
        # The max_length is based on https://github.com/THU-KEG/WaterBench/blob/main/config/dataset2maxlen.json.
        # The prompt_template is referenced from https://github.com/THU-KEG/WaterBench/blob/main/config/dataset2prompt.json.
        "longform_qa": {
            "sample_path": ROOT_DIR+"submodules/WaterBench/data/WaterBench/2-1_longform_qa.jsonl",
            "max_length": 300,
            "min_length": 200,
            "prompt_template": "You are a helpful assistant, please answer the following question within 300 words:\n{context}\n{input}",
        },
        "multi_news": {
            "sample_path": ROOT_DIR+"submodules/WaterBench/data/WaterBench/4-1_multi_news.jsonl",
            "max_length": 512,
            "min_length": 256,
            "prompt_template": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        },
    }
    IELTS_CONFIG = {
        # The max_length is calculated by the following formula:
        # max(all_split_human_answers_word_lengths)*1.5 = 2335.5 < 2500
        "ielts": {
            "sample_path": None,
            "max_length": 2500,
            "min_length": 0,
            "prompt_template": "{context}"
        }
    }
    OVERALL_CONFIG = {
        "waterbench": WATERBENCH_CONFIG,
        "ielts": IELTS_CONFIG,
    }

    def __init__(self):
        def __get_dataset2key(key:str) -> dict:
            return {
                dataset_name: benchmark_config[dataset_name][key]
                for benchmark_config in self.OVERALL_CONFIG.values()
                for dataset_name in benchmark_config
            }
        self.dataset2maxlen = __get_dataset2key("max_length")
        self.dataset2minlen = __get_dataset2key("min_length")
        self.dataset2prompt = __get_dataset2key("prompt_template")

    def get_benchmark_config(self, benchmark_name:str, dataset_name:str) -> dict:
        return self.OVERALL_CONFIG[benchmark_name][dataset_name]

    def get_min_max_lengths(self, dataset_name:str) -> tuple[int, int]:
        return self.dataset2minlen[dataset_name], self.dataset2maxlen[dataset_name]

    def __extract_placeholders(self, prompt_template:str) -> list[str]:
        pattern = r"\{(.*?)\}"
        return re.findall(pattern, prompt_template)

    def __sample_to_filled_prompt_and_conversation(self, sample_df:pd.DataFrame, prompt_template:str) -> tuple[list[str], list[list[dict]]]:
        prompt_placeholders = self.__extract_placeholders(prompt_template)
        prompt_fillers = [
            {label: sample_df[label].iloc[i] for label in prompt_placeholders}
            for i in range(len(sample_df))
        ]
        filled_prompts = [prompt_template.format(**filler) for filler in prompt_fillers]
        conversations = [[{"role": "user", "content": input}] for input in filled_prompts]
        return filled_prompts, conversations

    def get_samples_and_conversations(
        self, 
        logger:logging.Logger, 
        config:dict, 
        return_filled_prompts:bool=False
        ) -> tuple[pd.DataFrame, list[list[dict]], list[str]] | tuple[pd.DataFrame, list[list[dict]]]:
        input_config = config.get('file', {}).get('input', {})
        benchmark_name, dataset_name = input_config['benchmark'], input_config['dataset']
        benchmark_config = self.get_benchmark_config(benchmark_name, dataset_name)

        use_custom_dataset = input_config.get('use_custom_dataset', False)
        input_file_path = os.path.join(
            os.environ['ROOT_DIR'],
            input_config['custom_dataset'] if use_custom_dataset \
                else benchmark_config["sample_path"],
        )
        logger.info(f"Loading samples from {input_file_path}")
        sample_df = (load_from_disk(input_file_path)).to_pandas() if use_custom_dataset \
            else pd.read_json(input_file_path, orient="records", lines=True)
        sample_df = sample_df.head(config.get('generation', {}).get('max_samples', None))
        prompt_template = self.dataset2prompt[dataset_name]

        filled_prompts, conversations = self.__sample_to_filled_prompt_and_conversation(sample_df, prompt_template)

        if return_filled_prompts:
            return sample_df, conversations, filled_prompts
        else:
            return sample_df, conversations


class Preprocessor():
    def __init__(self, generation_df:pd.DataFrame, generation_field:str):
        self.generation_df = generation_df
        self.generation_field = generation_field
        self.supported_preprocess = {
            "extract_answer_with_regex": self.__extract_answer_with_regex,
            "remove_input_from_generation": self.__remove_input_from_generation,
        }
        self.supported_outputs = {
            "flatten": self.__flatten_to_1D_list,
            "pick_first_item": self.__pick_first_item,
            "as_is": self.__as_is,
        }


    def preprocess(self, preprocess_name:str, **kwargs) -> pd.DataFrame:
        if preprocess_name not in self.supported_preprocess:
            raise ValueError(f"Unsupported preprocess: {preprocess_name}")
        return self.supported_preprocess[preprocess_name](**kwargs)

    
    def __call__(self, preprocess_name:str, **kwargs) -> pd.DataFrame:
        return self.preprocess(preprocess_name, **kwargs)


    def __remove_input_from_generation(self, input_field:str, use_only_last_n_chars:int=-1, do_strip:bool=True) -> str:
        """
        Remove the input from the generation.
        Example:
        - Target: "Hello. I am Tom. I am a student. I am five years old."
        - Input: "Hello. I am Tom."
        - Preprocessed: "I am a student. I am five years old."
        """
        def __proc(row):
            gen_or_gen_list, inp = row[self.generation_field], row[input_field]
            if use_only_last_n_chars is not None:
                inp = inp[-min(len(inp), use_only_last_n_chars):]
            is_list = isinstance(gen_or_gen_list, list)
            gen_list = gen_or_gen_list if is_list else [gen_or_gen_list]
            preprocessed_gen_list = []
            for gen in gen_list:
                first_pos = gen.find(inp)
                if first_pos == -1:
                    raise ValueError(f"Input `{inp}` not found in generation `{gen}`.")
                preprocessed_gen = gen[first_pos + len(inp):]
                preprocessed_gen = preprocessed_gen.strip() if do_strip else preprocessed_gen
                preprocessed_gen_list.append(preprocessed_gen)
            return preprocessed_gen_list if is_list else preprocessed_gen_list[0]

        if use_only_last_n_chars<0:
            raise ValueError(f"use_only_last_n_chars must be non-negative, but got {use_only_last_n_chars}.")
        preprocessed_generation_df = self.generation_df.copy()
        preprocessed_generation_df[self.generation_field] = preprocessed_generation_df.apply(__proc, axis=1)
        self.generation_df = preprocessed_generation_df


    def __extract_answer_with_regex(self, regex_pattern:str, **kwargs) -> list[str]:
        """
        Extract the answer from the generation using a regex.
        Example:
        - Target: 
        """
        def __proc(row):
            gen_or_gen_list = row[self.generation_field]
            is_list = isinstance(gen_or_gen_list, list)
            gen_list = gen_or_gen_list if is_list else [gen_or_gen_list]
            extracted_list = []
            for gen in gen_list:
                extracted = re.search(regex_pattern, gen, flags=re.DOTALL)
                if extracted is None:
                    raise ValueError(f"Failed to extract answer with regex {regex_pattern} from {gen}.")
                extracted_list.append(extracted.group(1).strip())
            return extracted_list if is_list else extracted_list[0]
        preprocessed_generation_df = self.generation_df.copy()
        preprocessed_generation_df[self.generation_field] = preprocessed_generation_df.apply(__proc, axis=1)
        self.generation_df = preprocessed_generation_df


    def get_preprocessed_generations(self, method:str="as_is", **kwargs) -> list | pd.DataFrame:
        if method not in self.supported_outputs:
            raise ValueError(f"Unsupported extract method: {method}")
        return self.supported_outputs[method](**kwargs)
        

    def __flatten_to_1D_list(self, to_list:bool=True, **kwargs) -> list:
        if not to_list:
            raise ValueError(f"flatten only supports to_list=True, but got {to_list}.")
        if not all(isinstance(gen_list, list) for gen_list in self.generation_df[self.generation_field]):
            raise ValueError(f"Generation field {self.generation_field} must be a list of lists.")
        flatten_generation_list= [
            item \
            for generations in self.generation_df[self.generation_field] \
            for item in generations
            ]
        return flatten_generation_list


    def __pick_first_item(self, to_list:bool=True, **kwargs) -> list | pd.DataFrame:
        def __proc(row):
            generations = row[self.generation_field]
            if not isinstance(generations, list) or not generations:
                raise ValueError(f"Generation {generations} must be a non-empty list.")
            return generations[0]
        preprocessed_generation_df = self.generation_df.copy()
        preprocessed_generation_df[self.generation_field] = preprocessed_generation_df.apply(__proc, axis=1)
        return preprocessed_generation_df[self.generation_field].tolist() \
                if to_list \
                else preprocessed_generation_df

    
    def __as_is(self, to_list:bool=True, **kwargs) -> list | pd.DataFrame:
        return self.generation_df[self.generation_field].tolist() \
                if to_list \
                else self.generation_df


def get_output_path(config:dict, extension:str=".jsonl", custom_prefix:str="", custom_postfix:str="") -> str:
    output_path = os.path.join(
        os.environ['ROOT_DIR'],
        config['file']['output']['path'],
        f"{custom_prefix}{os.environ['RUN_NAME']}{custom_postfix}{extension}",
    )
    return output_path


def save_results_as_jsonl(logger:logging.Logger, config:dict, contents:dict[str, list], base_df:pd.DataFrame|None=None) -> None:
    logger.info(f"Saving results")
    result_df = pd.DataFrame(contents) if base_df is None else base_df.join(pd.DataFrame(contents))

    output_path = get_output_path(config)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Results saved to {output_path}")


def load_existing_results(logger:logging.Logger, config:dict) -> pd.DataFrame:
    output_path = get_output_path(config)
    if not os.path.exists(output_path):
        return None
    return pd.read_json(output_path, orient="records", lines=True)