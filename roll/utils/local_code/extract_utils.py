# Copyright LiveCodeBench @ 2024,

import re


def extract_code_generation(model_output: str, model_type: str = "chat"):
    # modified from
    # outputlines = model_output.split('\n')
    # TODO: handle codellama
    if "<|begin_of_solution|>" in model_output:
        model_output = model_output.split("<|begin_of_solution|>")[-1].strip()
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1].strip()
    if model_type == "base":
        return model_output.strip()
    elif model_type == "chat":
        code_pattern = r"```(cpp|python|java)\s*\n*(.*?)```"
        extract_code = re.findall(code_pattern, model_output, re.DOTALL)
        if extract_code and len(extract_code) > 1:
            return extract_code[-1][1]
        else:
            return model_output.strip()
    else:
        raise ValueError(f"Invalid mode type: {model_type}")
