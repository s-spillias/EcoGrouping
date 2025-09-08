import json
import os
from scripts.ask_AI import ask_ai
import numpy as np
import time
import functools
import unicodedata
import re
import matplotlib as plt
import pandas as pd

from typing import Dict, List, Optional, Literal
from collections import OrderedDict
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


def sanitize_text(text):
    """Clean and sanitize text input to handle encoding and special characters"""
    if pd.isna(text):
        return ""
    
    try:
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' 
                      or char in '\n\t')
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace from ends
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Error sanitizing text: {str(e)}")
        # Return sanitized version of original text
        return re.sub(r'[^\x20-\x7E\n\t]', '', str(text))

def extract_json_array(text):
    import re
    match = re.search(r'\[\s*{.*?}\s*]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            print("JSON decoding failed:", e)
    return None

def load_checkpoint(result_file):
    """Load checkpoint if it exists."""
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            return json.load(f)
    return None

def save_results(result_file, data):
    """Save results to a file."""
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)

def merge_responses(papers_json, parsed_response_list, dataset_file=None):
    """Merge parsed responses with original paper data, using dataset-specific matching keys."""
    # print(papers_json)
    # print(parsed_response_list)
    # print(dataset_file)
    # Define matching keys based on dataset_file
    matching_key = 'id'  # default
    if any(opt in dataset_file for opt in ("Chagaris", "Geers", "Montero")):
        matching_key = 'taxon'
    
    temp_dict = {d.get(matching_key, d.get('id')): d for d in papers_json}

    if parsed_response_list is None:
        print("Warning: parsed_response_list is None, returning original papers_json")
        return papers_json

    for item2 in parsed_response_list:
        try:
            item_id = item2.get(matching_key, item2.get('id', item2.get('paper_id')))
        except Exception as e:
            print(f"Error extracting ID: {e}")
            continue

        if item_id in temp_dict:
            temp_dict[item_id].update(item2)
        else:
            temp_dict[item_id] = item2

    return list(temp_dict.values())


    


def build_prompt(paper_or_list, dataset_file):
    """
    Build the screening or classification prompt based on the dataset type.
    
    Args:
        criteria (dict): Screening criteria.
        paper_or_list (dict or list): Single paper dict or list of paper dicts.
        dataset (str): Type of dataset ('screening' or starts with 'Chagaris', 'Geers', 'Montero').
    
    Returns:
        str: Formatted prompt string.
    """
    paper_json = json.dumps(paper_or_list, indent=2)
    base_prompt = ""

    # Construct the file path
    file_path = dataset_file
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = set()
        for v in data.values():
            if isinstance(v, list):
                values.update(v)
            else:
                values.add(v)
        options = sorted(values)

        base_prompt = (
            f"You are classifying taxa into functional groups for an ecosystem model.\n"
            f"The available functional groups are:\n{', '.join(options)}\n\n"
            "You will be provided with a list of taxonomic names. For each taxon, assign it to one of the functional groups listed above.\n"
            "Each functional group should represent organisms with similar:\n"
            "- Trophic level and feeding behavior (e.g., primary producers, herbivores, predators)\n"
            "- Habitat utilization (e.g., pelagic, benthic, reef-associated)\n"
            "- Body size and life history strategies\n"
            "- Ecological roles and energy transfer pathways\n"
            "- Vulnerability to fishing or environmental pressures\n\n"

            "Respond ONLY with a JSON array of objects in the following format:\n"
            "[\n"
            "  {\n"
            "    \"taxon\": \"Taxon Name\",\n"
            "    \"group\": \"Functional Group Name\",\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
        )
    else:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return base_prompt + "Here is the list items to classify:\n" + paper_json


def get_model_shorthand(model_name):
    """Flexible model shorthand mapping with three matching strategies:
    1. Exact match
    2. Substring match (key in model_name)
    3. Reverse substring match (model_name in key)
    """
    shorthand_map = {
        "claude-sonnet-4-20250514": "Claude-Sonnet-4",
        "qwen3:30b": "Qwen3-30B",
        "ollama_qwen3:235b-a22b-instruct-2507-q4_K_M": "Qwen3-235B",
        "qwen3:235b": "Qwen3-235B",
        "deepseek-r170b": "DeepSeek-R1-70B",
        "deepseek-r114b":"DeepSeek-R1-14B",
        "ollama_deepseek-r1:70b": "DeepSeek-R1-70B",
        "ollama_gpt-oss:latest": "GPT-oss-20B",
        "gpt-osslatest": "GPT-oss-20B",
        "gpt-oss": "GPT-oss-20B",
        "gpt-oss120b": "GPT-oss-120B",
        "ollama_gpt-oss:120b": "GPT-oss-120B",
        "ollama_gemma3n:latest": "Gemma3n",
        "gemma3n":"Gemma3n",
        "ollama_deepseek-r1:14b": "DeepSeek-R1-14B",
        "gemini-2.5-flash": "Gemini-Flash-2.5"
    }

    # 1. Exact match
    if model_name in shorthand_map:
        return shorthand_map[model_name]

    # 2. Substring match (key in model_name)
    for key, shorthand in shorthand_map.items():
        if key in model_name:
            return shorthand

    # 3. Reverse substring match (model_name in key)
    for key, shorthand in shorthand_map.items():
        if model_name in key:
            return shorthand

    # Fallback to original name
    return model_name



import hashlib
import hashlib
import re

def get_model_colors(model_names):
    """
    Assign hard-coded, publication-safe colors to model families,
    with brightness modulation based on model size.

    Parameters:
    - model_names: list of model names

    Returns:
    - Dictionary mapping model names to hex colors
    """
    family_color_map = {
        "deepseek-r1": "#005EB8",  # Blazer Blue
        "gpt-oss": "#009E73",      # Green
        "qwen3": "#CC79A7",        # Pink
        "claude": "#F47C5D",       # Orange
        "gemma3n": "#9B59B6",      # Purple
        "gemini": "#9B59B6",       # Purple (same as gemma)
    }

    def get_family(model_name):
        if "ollama_" in model_name:
            return model_name.split(":")[0].split("_")[1].lower()
        return model_name.split("-")[0].lower()

    def extract_size(model_name):
        match = re.search(r'(\d+)(b)', model_name.lower())
        if match:
            return int(match.group(1))
        return None

    def size_to_brightness(size):
        if size is None:
            return 0.8  # default
        elif size < 20:
            return 0.7
        elif size < 50:
            return 0.85
        else:
            return 1.0

    def adjust_color(hex_color, brightness):
        r = min(int(int(hex_color[1:3], 16) * brightness), 255)
        g = min(int(int(hex_color[3:5], 16) * brightness), 255)
        b = min(int(int(hex_color[5:7], 16) * brightness), 255)
        return f"#{r:02X}{g:02X}{b:02X}"

    color_map = {}
    for model_name in model_names:
        family = get_family(model_name)
        base_color = family_color_map.get(family, "#5D5D5D")  # Default gray
        size = extract_size(model_name)
        brightness = size_to_brightness(size)
        adjusted_color = adjust_color(base_color, brightness)
        color_map[model_name] = adjusted_color

    return color_map



def build_dataset_color_map(
    *,
    df: Optional[pd.DataFrame] = None,
    metrics_by_dataset: Optional[dict] = None,
    palette: str = "tab20",
    min_colors: int = 3
) -> Dict[str, tuple]:
    """
    Return a deterministic mapping {dataset_name: RGB_tuple}.

    The mapping is stable across runs as long as the set of dataset names is the
    same, because we sort the names before assigning palette colors.
    """
    import seaborn as sns

    names = set()
    if df is not None and not df.empty and ("dataset" in df.columns):
        names.update(df["dataset"].astype(str).unique().tolist())
    if isinstance(metrics_by_dataset, dict):
        names.update(str(k) for k in metrics_by_dataset.keys())

    names = sorted(names)
    if not names:
        return {}

    pal = sns.color_palette(palette, n_colors=max(min_colors, len(names)))
    return {name: pal[i % len(pal)] for i, name in enumerate(names)}
