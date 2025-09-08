import json
from scripts.ask_AI import ask_ai  # Replace 'your_module_name' with the actual module name
from dotenv import load_dotenv
import os
import sys
import random

load_dotenv()

def load_taxon_keys(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return list(data.keys())

def merge_and_save_output(output_file, all_responses):
    import json
    import os

    # Re-read the latest version of the file before saving
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8-sig') as file:
            try:
                latest_data = json.load(file)
            except json.JSONDecodeError:
                print("Warning: Could not decode existing output file. Proceeding with current data.")
                latest_data = {}
    else:
        latest_data = {}

    # Merge current all_responses into latest_data
    for taxon_key, model_data in all_responses.items():
        if taxon_key not in latest_data:
            latest_data[taxon_key] = {}
        for model, responses in model_data.items():
            if model not in latest_data[taxon_key]:
                latest_data[taxon_key][model] = []
            existing_replicates = len(latest_data[taxon_key][model])
            new_replicates = responses[existing_replicates:]
            latest_data[taxon_key][model].extend(new_replicates)

    # Save the merged result
    with open(output_file, 'w', encoding='utf-8-sig') as file:
        json.dump(latest_data, file, indent=4)

    print(f"Merged and saved output to '{output_file}'.")


def create_prompt(taxon_keys):
    template = (
    "Given the following taxonomic designations for an ecosystem model:\n"
    f"{taxon_keys}\n\n"
    "Create functional groups that prioritize ecological function over taxonomic classification. "
    "Each functional group should represent organisms with similar:\n"
    "- Trophic level and feeding behavior (e.g., primary producers, herbivores, predators)\n"
    "- Habitat utilization (e.g., pelagic, benthic, reef-associated)\n"
    "- Body size and life history strategies\n"
    "- Ecological roles and energy transfer pathways\n"
    "- Vulnerability to fishing or environmental pressures\n\n"
    "- Taxa/species of commercial or cultural importance should comprise their own group\n\n"

    "Guidelines for functional groups:\n"
    "- Aggregate taxa with similar P/B ratios (production to biomass)\n"
    "- Group organisms that occupy similar positions in food webs\n"
    "- Consider taxa with comparable Q/B ratios (consumption to biomass)\n"
    "- Separate commercially important taxa from ecologically similar but non-commercial ones\n"
    "- Balance model complexity with data availability\n"
    "- Ensure groups are meaningful for ecosystem management and policy questions\n\n"
    "Respond in the following JSON format without any additional text:\n"
    "{\n"
    "  \"functional_groups\": [\n"
    "    {\n"
    "      \"group_name\": \"string\",\n"
    "      \"description\": \"string describing ecological function, trophic role, and key characteristics\",\n"
    "      \"taxon\": \"[list of taxonomic names in this group]\",\n"

    "    },\n"
    "    ...\n"
    "  ]\n"
    "}"
)
    return template


def extract_json_from_response(response):
    """Extract the first JSON object from the response string."""
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end != -1:
        return response[start:end]
    return None

def get_valid_response(prompt, model, retries=3):
    for attempt in range(retries):
        response = ask_ai(prompt, model)
        extracted = extract_json_from_response(response)
        
        if extracted is None:
            print(f"Attempt {attempt + 1} failed: Extracted response is None.")
            continue
        
        try:
            parsed_response = json.loads(extracted)
            return parsed_response
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1} failed: Response is not valid JSON.")
    
    raise ValueError("Failed to get a valid JSON response after multiple attempts.")



def main(check=False):
    replicate = 5  # Set the number of replicates here
    dataset_path = 'Datasets/Groupings'
    taxon_files = [os.path.join(root, f) for root, _, files in os.walk(dataset_path) for f in files]
    all_models = [
        "claude-sonnet-4-20250514",
        # "ollama_hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL",
        # "ollama_gemma3n:latest",
        # "ollama_deepseek-r1:14b",
        "gemini-2.5-flash",
        # "ollama_deepseek-r1:70b",
        # "ollama_gpt-oss:latest",
        # "ollama_gpt-oss:120b",
        # "ollama_qwen3:235b-a22b-instruct-2507-q4_K_M"
    ]

    # Check for model name argument
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        models = [model_arg]
        print(f"Using model from command-line argument: {model_arg}")
    else:
        models = all_models

    output_file = 'Artifacts/Group/Output-Raw.json'

    # Load existing output if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8-sig') as file:
            all_responses = json.load(file)
    else:
        all_responses = {}

    for taxon_file in taxon_files:
        taxon_key = os.path.basename(taxon_file)
        if taxon_key not in all_responses:
            all_responses[taxon_key] = {}

        taxon_keys = load_taxon_keys(taxon_file)
        

        for model in models:
            if model not in all_responses[taxon_key]:
                all_responses[taxon_key][model] = []

            existing_replicates = len(all_responses[taxon_key][model])
            remaining_replicates = replicate - existing_replicates

            if remaining_replicates <= 0:
                print(f"Skipping {taxon_key} with model {model} (already has {replicate} replicates).")
                continue

            for i in range(remaining_replicates):
                print(f"Processing {taxon_key} with model {model}, replicate {existing_replicates + i + 1}...")
                random.shuffle(taxon_keys)
                prompt = create_prompt(taxon_keys)
                try:
                    if check:
                        print('NOT DONE')
                        continue
                    else:
                        valid_response = get_valid_response(prompt, model)
                except Exception as e:
                    print(f"Error during response generation: {e}")
                    valid_response = {
                        "error": str(e),
                        "placeholder": True,
                        "replicate": existing_replicates + i + 1,
                        "taxon_key": taxon_key,
                        "model": model
                    }

                all_responses[taxon_key][model].append(valid_response)

                # Save after each response
                merge_and_save_output(output_file, all_responses)

                print(f"Saved replicate {existing_replicates + i + 1} for {taxon_key} with model {model}.")

    print(f"All functional group proposals saved to '{output_file}'.")

if __name__ == "__main__":
    main(check=False)
