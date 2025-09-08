import pandas as pd
import re
import os
import numpy as np
from dotenv import load_dotenv
import random
import json
import tempfile
import shutil
from scripts.helper import (
    extract_json_array,
    build_prompt,
    retry
)
from scripts.ask_AI import ask_ai

import sys
load_dotenv()

# from filelock import FileLock,Timeout



@retry(max_attempts=3, delay=0.5)
def process_bulk_evaluation(papers_json, model, dataset_file):
    """Process bulk evaluation with checkpointing."""
    # combined_list_priority = load_checkpoint(result_file)
    chunk_size = len(papers_json)
    required_keywords = ["group","classification"
        ]
    # if combined_list_priority:
    #     print(f"  Loaded from checkpoint: {result_file}")
    #     return combined_list_priority

    prompt = build_prompt(papers_json,dataset_file)
    chunk_results = None
    response_num = 0
    while response_num < 3:
        response = ask_ai(prompt, model)

        # keyword_count = sum(response.count(keyword) for keyword in required_keywords)
        
        response_text = response[0] if isinstance(response, tuple) else response
        response_text = response_text.replace('taxa','taxon')
        print(response_text)
        chunk_results = extract_json_array(response_text)
        # Count keywords across all values in all dicts
        try:
            keyword_count = sum(
                k in required_keywords
                for item in chunk_results
                for k in item.keys()
            )

        except:
            keyword_count = 0
        print(f'checking chunk_size: {chunk_size}')
        print(f'keyword count: {keyword_count}')

        if keyword_count == chunk_size:
            break
        else:
            print('wrong format...trying again')
            response_num += 1
    if chunk_results is None:
        print("Failed to extract JSON array from response.")
        return papers_json
    return chunk_results
def create_paper_datasets(
    dataset_file,
    iterations_per_size,
    chunk_sizes=None,
    model='claude-sonnet-4-20250514',
    check_completion=False,
    # --- NEW TUNABLES ---
    min_completion_rate=0.85,         # minimum fraction of usable results per chunk
    max_failed_chunks=1,              # how many failed chunks allowed per chunk size
    stop_entire_run_on_breakdown=True # True = stop whole run; False = skip this chunk size
):
    """
    Process taxon dataset in chunks with separate result files per model.
    Adds health checks to fail fast if a chunk size is not performing.
    """
    with open(dataset_file, 'r', encoding='utf-8-sig') as f:
        input_raw = json.load(f)

    results_dir = "Artifacts/Classify/results"
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]

    # Create separate result file for this model
    model_safe = model.replace('/', '_').replace(':', '_')  # Make filename safe
    result_file = os.path.join(results_dir, f'{dataset_name}_{model_safe}_results.json')
    os.makedirs(results_dir, exist_ok=True)

    all_taxon_names = list(input_raw.keys())

    def load_existing_data():
        """Load existing results for this model."""
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading existing data: {e}. Starting fresh.")
                return []
        return []

    def save_results_local(result_data):
        """Save results for this model."""
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def get_processed_taxon_for_iteration(result_data, iteration_m, chunk_size):
        """Get set of taxon already processed for a specific iteration and chunk size."""
        existing_block = next(
            (entry for entry in result_data
             if (entry.get('metadata', {}).get('M') == iteration_m and
                 entry.get('metadata', {}).get('C') == chunk_size)),
            None
        )
        if existing_block:
            return {entry['taxon'] for entry in existing_block.get('content', []) if 'taxon' in entry}
        return set()

    # ---- NEW: helper to validate result objects ----
    def is_usable_result(r):
        # "Usable" = has a taxon and a non-missing group
        return isinstance(r, dict) and ('taxon' in r) and (r.get('group') not in (None, '', 'missing'))

    print(f"Using result file: {result_file}")

    for chunk_size in chunk_sizes:
        print(f"\nEvaluating list size C={chunk_size} with M={iterations_per_size} iterations...")

        # Adjust chunk size if needed
        if chunk_size > len(all_taxon_names):
            chunk_size = len(all_taxon_names)

        input_json = [{"taxon": taxon} for taxon in all_taxon_names]

        # --- NEW: counters for breakdown detection at this chunk size ---
        failed_chunks_at_this_size = 0
        breakdown_triggered = False

        for m in range(iterations_per_size):
            if breakdown_triggered:
                break  # stop additional iterations for this chunk size

            print(f"\n--- Starting iteration {m+1}/{iterations_per_size} for chunk size {chunk_size} ---")

            # Load existing data
            result_data = load_existing_data()
            processed_taxon = get_processed_taxon_for_iteration(result_data, m, chunk_size)

            # Check if iteration is complete
            if len(processed_taxon) >= len(set(all_taxon_names)):
                print(f"Iteration M={m}, C={chunk_size} already complete ({len(processed_taxon)} taxon). Skipping...")
                continue

            # Calculate remaining work
            remaining_taxon = set(all_taxon_names) - processed_taxon
            if not remaining_taxon:
                print(f"No remaining taxon for iteration M={m}, C={chunk_size}. Skipping...")
                continue

            print(f"Processing {len(remaining_taxon)} remaining taxon in chunks of {chunk_size}")
            random.shuffle(input_json)

            # Prepare chunks that need processing
            new_results = []
            chunks_to_process = []
            for i in range(0, len(input_json), chunk_size):
                chunk = input_json[i:i+chunk_size]
                chunk_to_process = [entry for entry in chunk if entry['taxon'] not in processed_taxon]
                if chunk_to_process:
                    chunks_to_process.append((i//chunk_size + 1, chunk_to_process))

            # Process each chunk
            total_chunks = len(chunks_to_process)
            for chunk_idx, chunk_data in chunks_to_process:
                print(f"Processing chunk {chunk_idx}/{(len(input_json) + chunk_size - 1)//chunk_size} ({len(chunk_data)} taxon)...")
                expected = len(chunk_data)
                got = 0

                try:
                    if check_completion:
                        print('NOT DONE')
                        continue

                    chunk_results = process_bulk_evaluation(chunk_data, model, dataset_file)
                    # Evaluate usable results immediately to detect breakdowns quickly
                    got = sum(1 for r in chunk_results if is_usable_result(r))

                    new_results.extend(chunk_results)
                    print(f"Chunk {chunk_idx} completed: usable={got}/{expected} ({got/expected:.1%})")

                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    got = 0  # treat as complete failure

                # --- NEW: failure check per chunk ---
                completion_rate = got / expected if expected else 0.0
                if completion_rate < min_completion_rate:
                    failed_chunks_at_this_size += 1
                    print(f"WARNING: Low completion at C={chunk_size}, M={m}, chunk {chunk_idx}: "
                          f"{completion_rate:.1%} < {min_completion_rate:.0%} "
                          f"(failed chunks so far: {failed_chunks_at_this_size}/{max_failed_chunks})")

                    if failed_chunks_at_this_size > max_failed_chunks:
                        print(f"\n### BREAKDOWN DETECTED at chunk size C={chunk_size}. "
                              f"Exceeded max_failed_chunks={max_failed_chunks}.")
                        breakdown_triggered = True
                        break  # stop processing more chunks for this size

            # Merge and save results (save what we have, even if breakdown occurred)
            if new_results:
                print(f"Merging {len(new_results)} new results...")
                # Reload data in case something else modified it
                result_data = load_existing_data()

                # Remove existing block for this iteration and chunk size
                result_data = [entry for entry in result_data
                               if not (entry.get('metadata', {}).get('M') == m and
                                       entry.get('metadata', {}).get('C') == chunk_size)]

                # Get any existing content for this iteration/chunk size combination
                existing_content = []
                for entry in result_data:
                    metadata = entry.get('metadata', {})
                    if metadata.get('M') == m and metadata.get('C') == chunk_size:
                        existing_content = entry.get('content', [])
                        break

                # Merge results with deduplication
                taxon_seen = {entry['taxon'] for entry in existing_content if 'taxon' in entry}
                merged_results = existing_content.copy()
                for result in new_results:
                    taxon = result.get('taxon')
                    if taxon and taxon not in taxon_seen:
                        taxon_seen.add(taxon)
                        if 'group' not in result:
                            result['group'] = 'missing'
                        merged_results.append(result)

                # Create metadata
                metadata = {
                    "dataset": dataset_name,
                    "model": model,
                    "C": chunk_size,
                    "M": m,
                    "total_taxon_processed": len(merged_results)
                }

                # Add new block
                result_data.append({
                    "filename": os.path.basename(result_file),
                    "metadata": metadata,
                    "content": merged_results
                })

                # Save results
                if save_results_local(result_data):
                    print(f"Successfully saved {len(merged_results)} total results for iteration M={m}, C={chunk_size}")
                else:
                    print(f"Failed to save results for iteration M={m}, C={chunk_size}")
            else:
                print(f"No new results to save for iteration M={m}, C={chunk_size}")

            # if breakdown detected after saving, decide whether to stop everything or skip size
            if breakdown_triggered:
                if stop_entire_run_on_breakdown:
                    print("\n*** Stopping ENTIRE RUN due to breakdown. ***")
                    return result_file
                else:
                    print("\n*** Skipping to next chunk size due to breakdown. ***")
                    break  # exit iteration loop; continue to next chunk size

    print(f"\nAll iterations completed! Results saved to: {result_file}")
    return result_file



if __name__ == "__main__":
    default_models = [
        #"ollama_deepseek-r1:70b",
        #"ollama_gpt-oss:latest",
        #"ollama_gpt-oss:120b",
        #"ollama_qwen3:235b-a22b-instruct-2507-q4_K_M",
        # "ollama_gemma3n:latest",
        #"ollama_deepseek-r1:14b",
        # "ollama_qwen3:30b-a3b-instruct-2507-q4_K_M",
        "claude-sonnet-4-20250514",
        # "gemini-2.5-flash",
    ]
    # print('run')

    folder_path = "Datasets"
    datasets = [os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files]

    expanded_chunk_sizes = [1, 2, 4, 8, 16, 32] + list(range(64, 129, 8)) + [256]
        # -> [1,2,4,8,16,32,64,72,80,88,96,104,112,120,128,256]

    # Check for command-line argument
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = default_models

    for model in models:
        print("#" * 50)
        print(f"Evaluating {model}.....")
        print("#" * 50)
        for dataset_file in datasets:
            print("#" * 50)
            print(f'Starting dataset: {dataset_file}')
            print("#" * 50)
            create_paper_datasets(
                dataset_file=dataset_file,
                iterations_per_size=5,
                chunk_sizes=expanded_chunk_sizes,
                #chunk_sizes=[10],
                model=model,
                check_completion=False
            )
