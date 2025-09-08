import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from scripts.helper import get_model_shorthand, get_model_colors
import csv
from statistics import mean
from collections import defaultdict
from difflib import get_close_matches
from sklearn.metrics.cluster import contingency_matrix
import warnings
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50% of the number of samples")
# from scripts.stats_collapse_test import run_threshold_monotonicity_tests, build_model_fit_results_from_summary, plot_metrics_by_model_with_fits, run_decline_after_one_tests
from scripts.stats_model import run_model_selection_with_bootstrap_and_plot, build_model_fit_results_from_summary
# #

# The sup‑F with unknown τ is the textbook way to handle a break parameter that’s not identified under H₀ (Andrews, 1993; Hansen, 2000). 13
# The wild cluster bootstrap (Rademacher) is a robust way to get finite‑sample p‑values with clustered data (e.g., few replicates per model). See MacKinnon (2015) and related literature; software often defaults to Rademacher weights. 56

def log_fuzzy_match(candidate_id, matched_id, predicted_group, json_path='Artifacts/Classify/Fuzzy.json'):
    """
    Log a fuzzy match to JSON file, ensuring only unique candidate_id/matched_id pairs are logged.

    Args:
        candidate_id (str): The original candidate ID that was fuzzy matched
        matched_id (str): The item ID it was matched to
        predicted_group (str): The predicted group
        json_path (str): Path to the JSON file (default: 'Artifacts/Classify/Fuzzy.json')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing data or initialize empty list
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            fuzzy_matches = json.load(f)
    else:
        fuzzy_matches = []

    # Check if the pair already exists
    # The any() function provides an efficient way to check for existence
    # without iterating through the whole list if a match is found early.
    is_duplicate = any(
        match['candidate_id'] == candidate_id and match['matched_id'] == matched_id
        for match in fuzzy_matches
    )

    # If it's not a duplicate, create and append the new entry
    if not is_duplicate:
        fuzzy_match_entry = {
            "candidate_id": candidate_id,
            "matched_id": matched_id
            # "predicted_group": predicted_group,
            # "timestamp": datetime.now().isoformat()
        }
        fuzzy_matches.append(fuzzy_match_entry)

        # Save the updated list to the file
        with open(json_path, 'w') as f:
            json.dump(fuzzy_matches, f, indent=2)

def robust_adjusted_rand_score(labels_true, labels_pred):
    contingency = contingency_matrix(labels_true, labels_pred)
    n = np.sum(contingency)
    P = sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1)) / 2
    Q = sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1)) / 2
    N = n * (n - 1) / 2

    # Compute Aw_i for each reference cluster
    Aw_list = []
    for i in range(contingency.shape[0]):
        ni_plus = np.sum(contingency[i, :])
        Pi = ni_plus * (ni_plus - 1) / 2
        if Pi == 0:
            continue
        T_i = sum(contingency[i, j] * (contingency[i, j] - 1) / 2 for j in range(contingency.shape[1]))
        Aw_i = (N * T_i - Pi * Q) / (Pi * (N - Q)) if (Pi * (N - Q)) != 0 else 0
        Aw_list.append(Aw_i)

    # Compute Av_j for each predicted cluster
    Av_list = []
    for j in range(contingency.shape[1]):
        n_plus_j = np.sum(contingency[:, j])
        Qj = n_plus_j * (n_plus_j - 1) / 2
        if Qj == 0:
            continue
        T_j = sum(contingency[i, j] * (contingency[i, j] - 1) / 2 for i in range(contingency.shape[0]))
        Av_j = (N * T_j - Qj * P) / (Qj * (N - P)) if (Qj * (N - P)) != 0 else 0
        Av_list.append(Av_j)

    # Compute robust adjusted Rand index AR†
    AW_dagger = np.mean(Aw_list)
    AV_dagger = np.mean(Av_list)
    AR_dagger = (2 * AW_dagger * AV_dagger) / (AW_dagger + AV_dagger) if (AW_dagger + AV_dagger) != 0 else 0

    return AR_dagger


# Functions to handle result files

def read_result_files(results_dir="results"):
    """
    Read all result files from the directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        dict: Dictionary mapping filenames to their content
    """
    result_files = {}
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return result_files
    
    # Read all JSON files in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                result_files[filename] = content
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    print(f"Read {len(result_files)} result files.")
    return result_files

def generate_qualitative_summary(analysis_results, output_path="Artifacts/Classify/Results-Qualitative.json"):
    summary_by_dataset = defaultdict(lambda: defaultdict(dict))
    dataset_valid_groups = {}
    dataset_dir = "Datasets/Groupings"

    if os.path.exists(dataset_dir):
        for fname in os.listdir(dataset_dir):
            if fname.endswith(".json"):
                dataset_name = fname.replace(".json", "")
                try:
                    with open(os.path.join(dataset_dir, fname), "r") as f:
                        data = json.load(f)
                        dataset_valid_groups[dataset_name] = set(data.values())
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

    for key, result in analysis_results.items():
        dataset = result.get("dataset", "Unknown")
        model = result.get("model", "Unknown")
        items = result["analysis"].get("items", [])
        if result['C'] != 1:
            continue
        for item in items:
            taxon_id = item.get('id') or item.get("taxon") or item.get("taxa")
            if taxon_id is None:
                continue

            taxon_data = summary_by_dataset[dataset].setdefault(taxon_id, {
                "total_attempts": 0,
                "total_correct": 0,
                "accuracy_by_model": defaultdict(lambda: {"correct": 0, "total": 0}),
                "mistaken_labels": {},
                "missing_by_model": defaultdict(int),
                "gold_standard": item.get("human_group") or item.get("human_classification")
            })
            # breakpoint()
            human_match = item.get("human_match")
            predicted_label = item.get("group") or item.get("classification")

            if human_match is not None:
                taxon_data["total_attempts"] += 1
                taxon_data["accuracy_by_model"][model]["total"] += 1
                if human_match:
                    taxon_data["total_correct"] += 1
                    taxon_data["accuracy_by_model"][model]["correct"] += 1
                else:
                    if predicted_label == "missing":
                        taxon_data["missing_by_model"][model] += 1
                    else:
                        if predicted_label not in taxon_data["mistaken_labels"]:
                            taxon_data["mistaken_labels"][predicted_label] = {}
                            taxon_data["mistaken_labels"][predicted_label]["is_valid_group"] = (
                                predicted_label in dataset_valid_groups.get(dataset, set())
                            )
                        taxon_data["mistaken_labels"][predicted_label].setdefault(model, 0)
                        taxon_data["mistaken_labels"][predicted_label][model] += 1

    final_summary = {}
    # breakpoint()
    for dataset, taxon_dict in summary_by_dataset.items():
        grouped_by_gold_standard = defaultdict(dict)
        for taxon_id, data in taxon_dict.items():
            total_attempts = data["total_attempts"]
            total_correct = data["total_correct"]
            overall_accuracy = total_correct / total_attempts if total_attempts > 0 else 0

            model_accuracy = {}
            missing_by_model = {}
            normalized_by_model = {}

            for model, info in data["accuracy_by_model"].items():
                total = info["total"]
                correct = info["correct"]
                accuracy = correct / total if total > 0 else 0
                missing = data["missing_by_model"].get(model, 0)
                missing_rate = missing / total if total > 0 else 0
                error = 1 - accuracy - missing_rate

                model_accuracy[model] = round(accuracy, 4)
                missing_by_model[model] = round(missing_rate, 4)
                normalized_by_model[model] = {
                    "accuracy": round(accuracy, 4),
                    "missing": round(missing_rate, 4),
                    "error": round(error, 4)
                }

            overall_missing = sum(data["missing_by_model"].values()) / total_attempts if total_attempts > 0 else 0
            overall_error = 1 - overall_accuracy - overall_missing

            normalized_overall = {
                "accuracy": round(overall_accuracy, 4),
                "missing": round(overall_missing, 4),
                "error": round(overall_error, 4)
            }

            gold_standard = data["gold_standard"] or "Unknown"
            # if gold_standard == "Unknown":
            grouped_by_gold_standard[gold_standard][taxon_id] = {
                "overall_accuracy": round(overall_accuracy, 4),
                "accuracy_by_model": model_accuracy,
                "missing_by_model": missing_by_model,
                "mistaken_labels": data["mistaken_labels"],
                "normalized_by_model": normalized_by_model,
                "normalized_overall": normalized_overall
            }
        final_summary[dataset] = grouped_by_gold_standard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_summary, f, indent=4)

import matplotlib.pyplot as plt

def shrink_xtick_font_if_overlaps(
    ax,
    base_size=14,
    min_size=6,
    step=1,
    rotation=45,
    ha='right',
    only_bottom_row=True,
    skip_if_few_ticks=0,  # e.g., set to 8 to never shrink when <= 8 ticks
    pad_px=1
):
    """
    Reset x-tick labels on `ax` to `base_size`, then shrink only if labels overlap.
    Run this AFTER layout is finalized (tight_layout or constrained layout draw).
    """

    # Optionally only adjust axes that display bottom labels (useful with sharex=True)
    if only_bottom_row:
        try:
            if hasattr(ax, "get_subplotspec") and not ax.get_subplotspec().is_last_row():
                # Still ensure the base styling on non-bottom rows if labels are shown
                for lab in ax.get_xticklabels():
                    lab.set_fontsize(base_size)
                    lab.set_rotation(rotation)
                    lab.set_horizontalalignment(ha)
                return
        except Exception:
            # If we can't determine row, proceed anyway
            pass

    # Reset styling to a known base before measuring overlap
    labels = ax.get_xticklabels()
    for lab in labels:
        lab.set_fontsize(base_size)
        lab.set_rotation(rotation)
        lab.set_horizontalalignment(ha)

    # Optionally skip shrinking when there are few ticks
    if skip_if_few_ticks and len([t for t in labels if t.get_text().strip()]) <= skip_if_few_ticks:
        return

    fig = ax.figure
    fig.canvas.draw()  # ensure up-to-date extents
    renderer = fig.canvas.get_renderer()

    labels = [lab for lab in ax.get_xticklabels() if lab.get_visible() and lab.get_text().strip()]
    if len(labels) <= 1:
        return

    # Overlap predicate using *actual* display-space bboxes
    def labels_overlap():
        bbs = [lab.get_window_extent(renderer=renderer) for lab in labels]
        # Sort by left edge
        bbs.sort(key=lambda b: b.x0)
        for a, b in zip(bbs[:-1], bbs[1:]):
            # Small padding to avoid "kissing" labels
            if a.x1 + pad_px >= b.x0 and a.overlaps(b):
                return True
        return False

    size = float(base_size)
    while size > min_size and labels_overlap():
        size -= step
        for lab in labels:
            lab.set_fontsize(size)
        fig.canvas.draw()  # force recompute before next check





def generate_accuracy_heatmap(json_path, output_path="Figures/analyse.png"):
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get dataset sizes for ordering
    dataset_info = []
    for dataset_name in data.keys():
        dataset_file = 'Datasets/Groupings/' + dataset_name + '.json'
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
                size_estimate = len(dataset_data)
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            size_estimate = 0
        
        dataset_info.append({
            "name": dataset_name,
            "size": size_estimate
        })
    
    # Sort datasets by size (largest first)
    dataset_info.sort(key=lambda x: x["size"], reverse=True)
    dataset_order = [info["name"] for info in dataset_info]
    print(f"Dataset order by size: {dataset_order}")
    
    # Prepare data for plotting
    records = []
    for dataset, groups in data.items():
        for group, taxon_data in groups.items():
            for taxon, metrics in taxon_data.items():
                norm_overall = metrics.get("normalized_overall", {})
                norm_by_model = metrics.get("normalized_by_model", {})
                
                # Overall metrics
                accuracy = norm_overall.get("accuracy", 0)
                missing = norm_overall.get("missing", 0)
                error = norm_overall.get("error", 0)
                
                # Normalize to sum to 1
                total = accuracy + missing + error
                if total > 0:
                    accuracy /= total
                    missing /= total
                    error /= total
                
                records.append({
                    "dataset": dataset,
                    "group": group,
                    "taxon": taxon,
                    "accuracy": accuracy,
                    "missing": missing,
                    "error": error,
                    "per_model": False
                })
                
                # Per-model metrics
                for model, values in norm_by_model.items():
                    acc = values.get("accuracy", 0)
                    miss = values.get("missing", 0)
                    err = values.get("error", 0)
                    
                    # Normalize to sum to 1
                    total = acc + miss + err
                    if total > 0:
                        acc /= total
                        miss /= total
                        err /= total
                    
                    records.append({
                        "dataset": dataset,
                        "group": group,
                        "taxon": taxon,
                        "model": model,
                        "accuracy": acc,
                        "missing": miss,
                        "error": err,
                        "per_model": True
                    })
    
    df = pd.DataFrame(records)
    df["per_model"] = df["per_model"].fillna(False)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Plot 1: Dataset-faceted plot with stacked bars split by model
    df_dataset = df[df["per_model"]].groupby(["dataset", "group", "model"]).agg({
        "accuracy": "mean",
        "missing": "mean", 
        "error": "mean"
    }).reset_index()
    
    # If no per-model data, fall back to overall data
    if df_dataset.empty:
        df_dataset = df[~df["per_model"]].groupby(["dataset", "group"]).agg({
            "accuracy": "mean",
            "missing": "mean", 
            "error": "mean"
        }).reset_index()
        df_dataset['model'] = 'Overall'
    
    # Create stacked bar plot for datasets (ordered by size)
    ordered_datasets = [d for d in dataset_order if d in df_dataset['dataset'].unique()]
    n_datasets = len(ordered_datasets)
    
    # Calculate dynamic width based on number of groups per dataset
    dataset_widths = []
    for dataset in ordered_datasets:
        n_groups = len(df_dataset[df_dataset['dataset'] == dataset]['group'].unique())
        # Base width of 1.5 per group, minimum of 3, maximum of 8
        width = max(3, min(8, 1.5 * n_groups + 1))
        dataset_widths.append(width)
    
    if n_datasets <= 3:
        nrows, ncols = 1, n_datasets
        figsize = (sum(dataset_widths), 6)
    else:
        ncols = 3
        nrows = (n_datasets + ncols - 1) // ncols  # Ceiling division
        # For multi-row layouts, use average width
        size_scalar = 0.9
        avg_width = size_scalar * sum(dataset_widths) / len(dataset_widths)
        figsize = (size_scalar * avg_width * ncols, size_scalar * 6 * nrows)
    
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single row case with dynamic widths
    if nrows == 1 and ncols > 1:
        # Create subplot with width ratios for single row
        fig1 = plt.figure(figsize=figsize)
        gs = fig1.add_gridspec(1, ncols, width_ratios=dataset_widths)
        axes1 = [fig1.add_subplot(gs[0, i]) for i in range(ncols)]
    elif nrows == 1 and ncols == 1:
        axes1 = [axes1]
    elif nrows == 1 or ncols == 1:
        axes1 = axes1.flatten()
    else:
        axes1 = axes1.flatten()
    
    for i, dataset in enumerate(ordered_datasets):
        data_subset = df_dataset[df_dataset['dataset'] == dataset]
        
        # Calculate overall accuracy per group for sorting
        group_accuracy = data_subset.groupby('group')['accuracy'].mean().reset_index()
        group_accuracy = group_accuracy.sort_values('accuracy', ascending=False)
        groups = group_accuracy['group'].tolist()  # Sorted by accuracy
        
        models = data_subset['model'].unique()
        n_models = len(models)
        model_height = 1.0 / n_models  # Each model gets equal segment of the bar
        
        group_positions = range(len(groups))
        
        # Use consistent colors for all models
        accuracy_color = '#2E8B57'  # Dark green
        missing_color = '#FFA500'   # Orange  
        error_color = '#DC143C'     # Dark red
        model_accuracies = {
            model: data_subset[data_subset['model'] == model]['accuracy'].mean()
            for model in models
        }
        sorted_models = sorted(models, key=lambda m: model_accuracies[m], reverse=True)


        for j, model in enumerate(sorted_models):
            model_data = data_subset[data_subset['model'] == model]
            
            # Calculate the base height for this model's segment
            segment_bottom = j * model_height
            
            # Prepare data arrays for each group
            accuracy_heights = []
            missing_heights = []
            error_heights = []
            
            for group in groups:
                group_data = model_data[model_data['group'] == group]
                if not group_data.empty:
                    acc = group_data['accuracy'].iloc[0]
                    miss = group_data['missing'].iloc[0]
                    err = group_data['error'].iloc[0]
                else:
                    acc = miss = err = 0
                
                # Scale the metrics to fit within this model's segment
                accuracy_heights.append(acc * model_height)
                missing_heights.append(miss * model_height)
                error_heights.append(err * model_height)
            
            # Create the stacked segments within this model's portion
            bottom_missing = [segment_bottom + acc for acc in accuracy_heights]
            bottom_error = [bot_miss + miss for bot_miss, miss in zip(bottom_missing, missing_heights)]
            
            # Only add labels for the first model to avoid legend clutter
            acc_label = 'Correct' if j == 0 else None
            miss_label = 'Missing' if j == 0 else None
            err_label = 'Incorrect' if j == 0 else None
            
            axes1[i].bar(group_positions, accuracy_heights, 
                        bottom=[segment_bottom] * len(groups),
                        label=acc_label, color=accuracy_color, alpha=1.0,
                        edgecolor='white', linewidth=0.5)
            axes1[i].bar(group_positions, missing_heights, 
                        bottom=bottom_missing,
                        label=miss_label, color=missing_color, alpha=1.0,
                        edgecolor='white', linewidth=0.5)
            axes1[i].bar(group_positions, error_heights, 
                        bottom=bottom_error,
                        label=err_label, color=error_color, alpha=1.0,
                        edgecolor='white', linewidth=0.5)
        
        # Add horizontal lines to separate model segments
        for k in range(1, n_models):
            axes1[i].axhline(y=k * model_height, color='black', linestyle='-', 
                           linewidth=1, alpha=0.7)
        
        # Add model labels on the right side
        for j, model in enumerate(sorted_models):
            y_pos = (j + 0.5) * model_height
            axes1[i].text(len(groups) - 0.4, y_pos, get_model_shorthand(model), 
                         rotation=0, ha='left', va='center', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        axes1[i].set_title(f'{dataset.replace('.json', '').replace('_', ' et al.\n')}', fontsize=12, fontweight='bold')
        axes1[i].set_ylabel('Proportion', fontsize=10)
        # axes1[i].set_xlabel('Group', fontsize=10)
        axes1[i].set_ylim(0, 1)
        
        # Set x-ticks and labels
        axes1[i].set_xticks(group_positions)
        axes1[i].set_xticklabels(groups, rotation=45, ha='right')  # no fontsize here



        
        # if i == 0:  # Only show legend on first subplot
        #     axes1[i].legend(loc='upper right', fontsize=9)
    
    # Hide extra subplots if we have more subplot positions than datasets
    for j in range(n_datasets, len(axes1)):
        axes1[j].set_visible(False)
    
    
    # fig1.suptitle("Normalized Metrics by Dataset (Stacked by Model)", fontsize=14, y=1.02)
    handles, labels = axes1[0].get_legend_handles_labels()

    # Position the legend to the far right outside all plots
    fig1.legend(handles, labels, 
            loc='center left',  # or 'upper left', 'lower left'
            bbox_to_anchor=(.95, 0.523),  # (x, y) - x=1.02 puts it outside the rightmost plot
            fontsize=14)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=1.0)  # Leave space for the legend

    # 🔽 ADD THIS BLOCK: shrink only where labels overlap
    fig1.canvas.draw()  # finalize geometry so bbox extents are accurate
    for ax in axes1:
        if getattr(ax, "get_visible", lambda: True)() and ax.get_visible():
            shrink_xtick_font_if_overlaps(
                ax,
                base_size=14,      # your normal label size
                min_size=6,       # don't go below this
                step=1,
                rotation=45,
                ha='right',
                only_bottom_row=False,  # you show x labels on all facets
                skip_if_few_ticks=8,    # never shrink if <= 8 ticks on this facet
                pad_px=2                # small safety margin
            )
    

    # Plot 2: Model-faceted plot with stacked bars  
    df_model = df[df["per_model"]].groupby(["model", "group"]).agg({
        "accuracy": "mean",
        "missing": "mean",
        "error": "mean"
    }).reset_index()
    
    if not df_model.empty:
        n_models = len(df_model['model'].unique())
        if n_models <= 3:
            nrows, ncols = 1, n_models
            figsize = (4 * ncols, 6)
        else:
            ncols = 3
            nrows = (n_models + ncols - 1) // ncols  # Ceiling division
            figsize = (4 * ncols, 6 * nrows)
        
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes2 = [axes2]
        elif nrows == 1 or ncols == 1:
            axes2 = axes2.flatten()
        else:
            axes2 = axes2.flatten()
        
        for i, model in enumerate(df_model['model'].unique()):
            data_subset = df_model[df_model['model'] == model]
            
            # Sort groups by accuracy (highest to lowest)
            data_subset = data_subset.sort_values('accuracy', ascending=False)
            
            # Create stacked bars
            bottom_missing = data_subset['accuracy']
            bottom_error = data_subset['accuracy'] + data_subset['missing']
            
            axes2[i].bar(data_subset['group'], data_subset['accuracy'], 
                        label='Accuracy', color='#2E8B57', alpha=1.0)
            axes2[i].bar(data_subset['group'], data_subset['missing'], 
                        bottom=bottom_missing, label='Missing', color='#FFA500', alpha=1.0)
            axes2[i].bar(data_subset['group'], data_subset['error'], 
                        bottom=bottom_error, label='Error', color='#DC143C', alpha=1.0)
            
            axes2[i].set_title(f'{model}', fontsize=12, fontweight='bold')
            axes2[i].set_ylabel('Proportion', fontsize=10)
            # axes2[i].set_xlabel('Group', fontsize=10)
            axes2[i].set_ylim(0, 1)
            
            # Set rotation/alignment first, not the size
            axes2[i].tick_params(axis='x', rotation=45)
            for label in axes2[i].get_xticklabels():
                label.set_ha('right')

            # (leave the rest of the plotting loop as is)

            # After you finish building all subplots and call tight_layout(), add:
            plt.tight_layout()

            # 🔽 ADD THIS BLOCK: shrink only where labels overlap
            fig2.canvas.draw()
            for ax in axes2:
                if getattr(ax, "get_visible", lambda: True)() and ax.get_visible():
                    shrink_xtick_font_if_overlaps(
                        ax,
                        base_size=14,
                        min_size=6,
                        step=1,
                        rotation=45,
                        ha='right',
                        only_bottom_row=False,
                        skip_if_few_ticks=8,
                        pad_px=2
                    )
            

            
            if i == 0:  # Only show legend on first subplot
                axes2[i].legend(loc='upper right', fontsize=9)
        
        # Hide extra subplots if we have more subplot positions than models
        for j in range(n_models, len(axes2)):
            axes2[j].set_visible(False)
        
        plt.tight_layout()
        # fig2.suptitle("Normalized Metrics by Model (Stacked)", fontsize=14, y=1.02)
    
    
    # Save the plots
    dataset_path = output_path.replace("analyse.png", "Figure_Group_Accuracy.png")
    model_path = output_path.replace("analyse.png", "Supp_Model_Accuracy.png")
    
    fig1.savefig(dataset_path, dpi=300, bbox_inches='tight')
    if not df_model.empty:
        fig2.savefig(model_path, dpi=300, bbox_inches='tight')
    
   
    print(f"Dataset plot saved to {dataset_path}")
    if not df_model.empty:
        print(f"Model plot saved to {model_path}")
    else:
        print("No per-model data found, skipping model plot")
    
    return fig1, fig2 if not df_model.empty else None


def save_analysis_results(analysis_results, 
                          json_filename="Artifacts/Classify/Results.json", 
                          csv_filename="Artifacts/Classify/Results-List.csv", 
                          summary_json_filename="Artifacts/Classify/Results-Summary.json"):
    # Save full results to a JSON file with reordered keys
    reordered_results = {}
    for key, result in analysis_results.items():
        reordered_analysis = {}
        analysis = result["analysis"]
        # Move metrics and confusion_matrix to the top
        if "metrics" in analysis:
            reordered_analysis["metrics"] = analysis["metrics"]
        if "confusion_matrix" in analysis:
            reordered_analysis["confusion_matrix"] = analysis["confusion_matrix"]
        for k, v in analysis.items():
            if k not in ("metrics", "confusion_matrix"):
                reordered_analysis[k] = v
        reordered_results[key] = {
            "file": result.get("file"),
            "C": result.get("C"),
            "N": result.get("N"),
            "M": result.get("M"),
            "model": result.get("model"),
            "analysis": reordered_analysis
        }

    with open(json_filename, 'w') as json_file:
        json.dump(reordered_results, json_file, indent=4)

    # Save summary to a CSV file with flattened metrics
    with open(csv_filename, 'w', newline='') as csv_file:
        fieldnames = [
            "file", "C", "N", "M", "model", "human_agreement", 
            "items_with_human_classification", 
            "macro_precision", "macro_recall", "macro_f1_score",
            "micro_precision", "micro_recall", "micro_f1_score",
            "recall", "precision", "f1_score", "specificity"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in analysis_results.values():
            metrics = result["analysis"].get("metrics", {})
            row = {
                "file": result.get("file"),
                "C": result.get("C"),
                "N": result.get("N"),
                "M": result.get("M"),
                "model": result.get("model"),
                "human_agreement": result["analysis"].get("human_agreement"),
                "items_with_human_classification": result["analysis"].get("items_with_human_classification"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "macro_f1_score": metrics.get("macro_f1_score"),
                "micro_precision": metrics.get("micro_precision"),
                "micro_recall": metrics.get("micro_recall"),
                "micro_f1_score": metrics.get("micro_f1_score"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
                "f1_score": metrics.get("f1_score"),
                "specificity": metrics.get("specificity")
            }
            writer.writerow(row)

    # Create summarized JSON grouped by model and C
    summary = defaultdict(lambda: defaultdict(list))
    for result in analysis_results.values():
        model = result.get("model")
        C = result.get("C")
        metrics = result["analysis"].get("metrics", {})
        summary[model][C].append(metrics)

    summarized_output = {}
    for model, c_dict in summary.items():
        summarized_output[model] = {}
        for C, metrics_list in c_dict.items():
            aggregated = {}
            for metric in metrics_list:
                for k, v in metric.items():
                    aggregated.setdefault(k, []).append(v)
            averaged = {k: mean(vs) for k, vs in aggregated.items()}
            summarized_output[model][C] = {
                "average_metrics": averaged,
                "num_runs": len(metrics_list)
            }

    # Add model-only summary to the same JSON
    model_only_summary = defaultdict(list)
    for result in analysis_results.values():
        model = result.get("model")
        metrics = result["analysis"].get("metrics", {})
        model_only_summary[model].append(metrics)

    summarized_output["model_only_summary"] = {}
    for model, metrics_list in model_only_summary.items():
        aggregated = {}
        for metric in metrics_list:
            for k, v in metric.items():
                aggregated.setdefault(k, []).append(v)
        averaged = {k: mean(vs) for k, vs in aggregated.items()}
        summarized_output["model_only_summary"][model] = {
            "average_metrics": averaged,
            "num_runs": len(metrics_list)
        }

    # Write to JSON file
    with open(summary_json_filename, 'w') as summary_file:
        json.dump(summarized_output, summary_file, indent=4)


    # print(f"Saved full results to '{json_filename}', summary CSV to '{csv_filename}', and summarized JSON to '{summary_json_filename}'.")

def load_human_classifications(human_labels_file):
    """
    Load human classifications from one or more files (CSV or JSON).
    
    Args:
        human_labels_files: List of paths to files containing human classifications
        
    Returns:
        dict: Mapping of item IDs to human classifications
    """
    human_classifications = []
    # print(human_labels_file)
    try:
        if human_labels_file.endswith(".csv"):
            human_labels = pd.read_csv(human_labels_file)

            if "Include or exclude" in human_labels.columns:
                human_labels['item_id'] = ["paper_" + str(i) for i in human_labels.index]
                human_labels['Include or exclude'] = human_labels['Include or exclude'].str.lower().map(
                    lambda x: 'include' if x == 'include' else 'not_include'
                )

                for _, row in human_labels.iterrows():
                    human_classifications.append({
                        'item_id': row['item_id'],
                        'classification': row['Include or exclude']
                    })

                # print(f"Loaded {len(human_classifications)} classifications from {human_labels_file}")

        elif "Groupings" in str(human_labels_file):
            with open(human_labels_file, 'r') as f:
                data = json.load(f)
                for taxon, group in data.items():
                    human_classifications.append({'taxon': taxon, 'group': group})
                # print(f"Loaded {len(data)} classifications from {human_labels_file}")

    except Exception as e:
        print(f"Error loading classifications from {human_labels_file}: {e}")

    return human_classifications

def plot_chunk_size_effect(chunk_effect_results, output_prefix="chunk_size_effect"):
    """
    Create plots visualizing the effect of chunk size on agreement metrics.
    
    Args:
        chunk_effect_results: Results from analyze_chunk_size_effect
        output_prefix: Prefix for output plot files
    """
    # Prepare data for plotting
    chunk_sizes = sorted(chunk_effect_results.keys())
    avg_agreements = [chunk_effect_results[c]["avg_agreement"] for c in chunk_sizes]
    
    # Check if human data is available
    human_data_available = all(chunk_effect_results[c]["avg_human_agreement_regular"] is not None for c in chunk_sizes)
    
    if human_data_available:
        avg_human_regular = [chunk_effect_results[c]["avg_human_agreement_regular"] for c in chunk_sizes]
        avg_human_individual = [chunk_effect_results[c]["avg_human_agreement_individual"] for c in chunk_sizes]
        improvements = [chunk_effect_results[c]["improvement"] for c in chunk_sizes]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Effect of chunk size on within-chunk agreement
    plt.subplot(2, 2, 1)
    plt.plot(chunk_sizes, avg_agreements, 'o-', linewidth=2)
    plt.xlabel('Batch Size (B)')
    plt.ylabel('Within-Batch Agreement')
    plt.title('Effect of Batch Size on Within-Batch Agreement')
    plt.grid(True, alpha=0.3)
    
    if human_data_available:
        # Plot 2: Effect of chunk size on agreement with human - regular
        plt.subplot(2, 2, 2)
        plt.plot(chunk_sizes, avg_human_regular, 'o-', linewidth=2)
        plt.xlabel('Batch Size (C)')
        plt.ylabel('Agreement with Human')
        plt.title('Effect of Batch Size on Agreement with Human - Regular')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Effect of chunk size on agreement with human - individual
        plt.subplot(2, 2, 3)
        plt.plot(chunk_sizes, avg_human_individual, 'o-', linewidth=2)
        plt.xlabel('Batch Size (C)')
        plt.ylabel('Agreement with Human')
        plt.title('Effect of Batch Size on Agreement with Human - Individual')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Effect of chunk size on improvement
        plt.subplot(2, 2, 4)
        plt.plot(chunk_sizes, improvements, 'o-', linewidth=2)
        plt.xlabel('Batch Size (C)')
        plt.ylabel('Improvement')
        plt.title('Effect of Batch Size on Improvement from Individual Treatment')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures/' + f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Batch size effect plot saved to '{output_prefix}.png'")

def analyze_human_agreement(content, human_classifications,filename):
    """
    Analyze agreement with human classifications if available.
    
    Args:
        content: Content of a result file
        human_classifications: Optional dictionary mapping item IDs to human classifications
        
    Returns:s
        dict: Analysis results focusing on agreement with human classifications
    """
    if not content or not human_classifications:
        return {
            "human_agreement": 0,
            "items_with_human_classification": 0,
            "items": [],
            "metrics": {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1_score": 0
            }
        }
    y_true = []
    y_pred = []
    items_results = []
    human_agreement_count = 0
    items_with_human_classification = 0

    
    # Load known taxon once
    with open(filename, 'r', encoding='utf-8') as f:
        all_dicts = json.load(f)

    item_possible = list(all_dicts.keys())
    # breakpoint()
    human_taxon_names = [d['taxon'] for d in human_classifications]


    for item_id in item_possible:
        # Find the predicted group from content
        predicted_group = ""
        for item in content:
            candidate_id = item.get("taxon") or item.get("speciees") or item.get("species")
            if candidate_id == item_id:
                predicted_group = item.get("group", "")
                break
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(candidate_id, [item_id], n=1, cutoff=0.9)
                if close_matches:
                    predicted_group = item.get("group", "")
                    log_fuzzy_match(candidate_id, item_id, predicted_group)  # Add this line
                    # print(f"Fuzzy matched '{candidate_id}' to '{item_id}'")
                    break

        # Try to find human classification
        try:
            human_group = next((d for d in human_classifications if d['taxon'] == item_id), None)["group"]
        except (TypeError, KeyError):
            human_group = None
            close_human_matches = get_close_matches(item_id, human_taxon_names, n=1, cutoff=0.9)
            if close_human_matches:
                matched_human_taxon = close_human_matches[0]
                try:
                    human_group = next((d for d in human_classifications if d['taxon'] == matched_human_taxon), None)["group"]
                    print(f"Fuzzy matched human classification '{item_id}' to '{matched_human_taxon}'")
                except (TypeError, KeyError):
                    human_group = None

        human_match = predicted_group == human_group if human_group else False

        if human_group:
            items_with_human_classification += 1
            y_true.append(human_group)
            y_pred.append(predicted_group)
            if human_match:
                human_agreement_count += 1

        items_results.append({
            "id": item_id,
            "group": predicted_group,
            "human_group": human_group,
            "human_match": human_match
        })

    human_agreement = human_agreement_count / items_with_human_classification if items_with_human_classification > 0 else 0

    # Compute macro-averaged metrics
# try:
    macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
# except:
#     breakpoint()
    # Compute micro-averaged metrics
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
)

    # Compute robust adjusted Rand score
    robust_ari = robust_adjusted_rand_score(y_true,y_pred)

    return {
        "human_agreement": human_agreement,
        "items_with_human_classification": items_with_human_classification,
        "items": items_results,
        "metrics": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1_score,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1_score": micro_f1_score,
            "robust_adjusted_rand_score": robust_ari
        }
    }

def calculate_metrics_by_chunk_size(analysis_results):
    """
    Calculate metrics for each combination of dataset, model, and chunk size, supporting both binary and multi-class classification.
    
    Args:
        analysis_results: Results from run_agreement_analysis
        
    Returns:
        dict: Nested dict with structure {dataset: {model: {chunk_size: metrics}}}
    """
    # Group by dataset, model and C (chunk size)
    dataset_model_c_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, result in analysis_results.items():
        dataset = result.get("dataset", "Unknown")
        model = result["model"]
        c = result["C"]
        dataset_model_c_groups[dataset][model][c].append(result)
    metrics_by_chunk_size = {}
    
    # Calculate metrics for each dataset, model and chunk size combination
    for dataset, model_groups in dataset_model_c_groups.items():
        metrics_by_chunk_size[dataset] = {}
        
        for model, c_groups in model_groups.items():
            metrics_by_chunk_size[dataset][model] = {}
            
            for C, results in sorted(c_groups.items()):
            # Get results with human classifications
                results_with_human = [r for r in results if r["analysis"].get("items_with_human_classification", 0) > 0]
                
                if results_with_human:
                    # Separate results by classification type since they can be mixed
                    binary_results = []
                    multiclass_results = []
                    
                    for r in results_with_human:
                        metrics = r["analysis"]["metrics"]
                        if "recall" in metrics and "specificity" in metrics:
                            binary_results.append(r)
                        elif "macro_precision" in metrics:
                            multiclass_results.append(r)
                    
                    chunk_metrics = {"num_results": len(results_with_human)}
                    
                    # Process binary classification results if any
                    if binary_results:
                    
                        try:
                            avg_recall = np.mean([r["analysis"]["metrics"]["recall"] for r in binary_results])
                            avg_precision = np.mean([r["analysis"]["metrics"]["precision"] for r in binary_results])
                            avg_f1_score = np.mean([r["analysis"]["metrics"]["f1_score"] for r in binary_results])
                            avg_specificity = np.mean([r["analysis"]["metrics"]["specificity"] for r in binary_results])
                            
                            # Calculate standard deviations
                            std_recall = np.std([r["analysis"]["metrics"]["recall"] for r in binary_results])
                            std_precision = np.std([r["analysis"]["metrics"]["precision"] for r in binary_results])
                            std_f1_score = np.std([r["analysis"]["metrics"]["f1_score"] for r in binary_results])
                            std_specificity = np.std([r["analysis"]["metrics"]["specificity"] for r in binary_results])
                            
                            chunk_metrics["binary"] = {
                                "num_results": len(binary_results),
                                "avg_metrics": {
                                    "recall": avg_recall,
                                    "precision": avg_precision,
                                    "f1_score": avg_f1_score,
                                    "specificity": avg_specificity
                                },
                                "std_metrics": {
                                    "recall": std_recall,
                                    "precision": std_precision,
                                    "f1_score": std_f1_score,
                                    "specificity": std_specificity
                                }
                            }
                        except KeyError as e:
                            print(f"KeyError in binary results for chunk size {C}: {e}")
                            raise
                    
                    # Process multi-class classification results if any
                    if multiclass_results:
                      
                        try:
                            avg_macro_precision = np.mean([r["analysis"]["metrics"]["macro_precision"] for r in multiclass_results])
                            avg_macro_recall = np.mean([r["analysis"]["metrics"]["macro_recall"] for r in multiclass_results])
                            avg_macro_f1_score = np.mean([r["analysis"]["metrics"]["macro_f1_score"] for r in multiclass_results])
                            
                            # Calculate standard deviations
                            std_macro_precision = np.std([r["analysis"]["metrics"]["macro_precision"] for r in multiclass_results])
                            std_macro_recall = np.std([r["analysis"]["metrics"]["macro_recall"] for r in multiclass_results])
                            std_macro_f1_score = np.std([r["analysis"]["metrics"]["macro_f1_score"] for r in multiclass_results])
                            
                            # Collect robust adjusted Rand (AR†) if present
                            ari_vals = [
                                r["analysis"]["metrics"].get("robust_adjusted_rand_score")
                                for r in multiclass_results
                                if r["analysis"]["metrics"].get("robust_adjusted_rand_score") is not None
                            ]
                            if ari_vals:
                                avg_ari = np.mean(ari_vals)
                                std_ari = np.std(ari_vals)
                            else:
                                avg_ari, std_ari = None, None

                            multiclass_chunk_metrics = {
                                "num_results": len(multiclass_results),
                                "avg_metrics": {
                                    "macro_precision": avg_macro_precision,
                                    "macro_recall":    avg_macro_recall,
                                    "macro_f1_score":  avg_macro_f1_score,
                                    # NEW: robust ARI mean
                                    "robust_adjusted_rand_score": avg_ari,
                                },
                                "std_metrics": {
                                    "macro_precision": std_macro_precision,
                                    "macro_recall":    std_macro_recall,
                                    "macro_f1_score":  std_macro_f1_score,
                                    # NEW: robust ARI std
                                    "robust_adjusted_rand_score": std_ari,
                                }
                            }

                            
                            # Check if micro metrics are also available
                            if "micro_precision" in multiclass_results[0]["analysis"]["metrics"]:
                                avg_micro_precision = np.mean([r["analysis"]["metrics"]["micro_precision"] for r in multiclass_results])
                                avg_micro_recall = np.mean([r["analysis"]["metrics"]["micro_recall"] for r in multiclass_results])
                                avg_micro_f1_score = np.mean([r["analysis"]["metrics"]["micro_f1_score"] for r in multiclass_results])
                                
                                std_micro_precision = np.std([r["analysis"]["metrics"]["micro_precision"] for r in multiclass_results])
                                std_micro_recall = np.std([r["analysis"]["metrics"]["micro_recall"] for r in multiclass_results])
                                std_micro_f1_score = np.std([r["analysis"]["metrics"]["micro_f1_score"] for r in multiclass_results])
                                
                                multiclass_chunk_metrics["avg_metrics"].update({
                                    "micro_precision": avg_micro_precision,
                                    "micro_recall": avg_micro_recall,
                                    "micro_f1_score": avg_micro_f1_score
                                })
                                multiclass_chunk_metrics["std_metrics"].update({
                                    "micro_precision": std_micro_precision,
                                    "micro_recall": std_micro_recall,
                                    "micro_f1_score": std_micro_f1_score
                                })
                            
                            chunk_metrics["multiclass"] = multiclass_chunk_metrics
                            
                        except KeyError as e:
                            print(f"KeyError in multiclass results for chunk size {C}: {e}")
                            raise
                    
                    metrics_by_chunk_size[dataset][model][C] = chunk_metrics
                else:
                    metrics_by_chunk_size[dataset][model][C] = {
                        "num_results": 0,
                        "binary": None,
                        "multiclass": None
                    }
            
    return metrics_by_chunk_size


def plot_metrics_by_model(metrics_by_dataset, output_prefix="metrics_by_model"):
    """
    Facet by MODEL. Within each subplot (one per model), plot macro-F1 vs batch size
    for every dataset, using consistent dataset colors and the same plotting
    conventions as plot_combined_metrics (categorical x-axis with string labels,
    error bars, grid, tight_layout, and a figure-level legend outside).
    """
    # ---- Collect dataset info (names, clean labels, sizes) ----
    dataset_names = sorted(metrics_by_dataset.keys())
    all_model_names = set()
    dataset_info = []

    for dataset_name in metrics_by_dataset:
        dataset_file = 'Datasets/Groupings/' + dataset_name + '.json'
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                size_estimate = len(data)
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            size_estimate = 0

        for model_name in metrics_by_dataset[dataset_name].keys():
            all_model_names.add(model_name)

        dataset_info.append({
            "name": dataset_name,
            # same cleaning as plot_combined_metrics
            "clean_name": re.sub(r'(?<!^)(?=[A-Z])', '', dataset_name.replace('.json', '').replace('_', '\n')).replace('Mcmullen','McMullen'),
            "size": size_estimate
        })

    # Order datasets by size (largest first), mirroring plot_combined_metrics
    dataset_info.sort(key=lambda x: x["size"], reverse=True)
    dataset_names = [info["name"] for info in dataset_info]
    clean_names = [info["clean_name"] for info in dataset_info]
    dataset_sizes = [info["size"] for info in dataset_info]

    # Models present
    model_names = sorted(all_model_names)

    # ---- Colors: dataset-centric palette (since lines represent datasets) ----
    # We mirror the idea of consistent palette usage from plot_combined_metrics
    # but apply it to datasets here.
    import seaborn as sns
    n_datasets = len(dataset_names)
    # Try tab20 then fall back to husl if more needed
    base_palette = sns.color_palette("Set2", n_datasets)  
    # if n_datasets > len(base_palette):
    #     base_palette = sns.color_palette("husl", n_datasets)
    dataset_color_map = {ds: base_palette[i % n_datasets] for i, ds in enumerate(dataset_names)}

    # Map dataset -> clean label for legend
    dataset_label_map = {
    info["name"]: f"{info['clean_name']} ({info['size']})"
    for info in dataset_info
}


    # ---- Figure layout: facet by model (same grid logic style) ----
    n_models = len(model_names)
    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        axes = [axes]
    elif n_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    elif n_models <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
    else:
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 3 * n_rows))
        axes = axes.flatten()
    all_chunk_sizes = set()
    for ds in dataset_names:
        ds_data = metrics_by_dataset.get(ds, {})
        for model_name in model_names:
            if model_name in ds_data:
                all_chunk_sizes.update(ds_data[model_name].keys())

    all_chunk_sizes = sorted(all_chunk_sizes)
    chunk_size_labels = [str(cs) for cs in all_chunk_sizes]
    chunk_size_positions = list(range(len(all_chunk_sizes)))
    # ---- Plot each model facet ----
    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]

        # Determine the union of chunk sizes available for this model across datasets
        model_chunk_sizes = set()
        for ds in dataset_names:
            ds_data = metrics_by_dataset.get(ds, {})
            if model_name in ds_data:
                model_chunk_sizes.update(ds_data[model_name].keys())
        model_chunk_sizes = sorted(model_chunk_sizes)
        # chunk_size_labels = [str(cs) for cs in model_chunk_sizes]
        # chunk_size_positions = list(range(len(model_chunk_sizes)))

        # Within this model, plot one line per dataset (colored by dataset)
        for ds in dataset_names:
            ds_data = metrics_by_dataset.get(ds, {})
            if model_name not in ds_data:
                continue

            model_data = ds_data[model_name]

            # Build y-values aligned to the unioned x positions (None where missing)
            f1_vals = []
            f1_errs = []
            for cs in all_chunk_sizes:
                # robust get for cs possibly being str or int in your dict
                entry = model_data.get(cs, model_data.get(str(cs)))
                if entry:
                    mc = entry.get("multiclass")
                    if mc and mc["avg_metrics"] is not None:
                        f1_vals.append(mc["avg_metrics"]["macro_f1_score"])
                        f1_errs.append(mc["std_metrics"]["macro_f1_score"])
                    else:
                        f1_vals.append(None)
                        f1_errs.append(None)
                else:
                    f1_vals.append(None)
                    f1_errs.append(None)


            # Filter None rows for plotting
            idxs = [i for i, y in enumerate(f1_vals) if y is not None]
            if not idxs:
                continue
            x_vals = idxs  # aligned with all_chunk_sizes
            y_vals = [f1_vals[i] for i in idxs]
            y_errs = [f1_errs[i] for i in idxs]


            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt='o-', linewidth=1, capsize=3, alpha=0.7, markersize=3,
                color=dataset_color_map[ds],
                label=dataset_label_map[ds]
            )

            # Axes styling to match plot_combined_metrics
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Macro F1 Score')
            ax.set_title(get_model_shorthand(model_name))
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(all_chunk_sizes)))
            ax.set_xticklabels([str(cs) for cs in all_chunk_sizes], rotation=45, ha='center', va='top', fontsize=8)
            # ax.grid(True, alpha=0.3)


    # Hide any unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # ---- Figure-level legend outside (consistent layout) ----
    # Build proxy handles so legend includes all datasets even if some missing in first subplot
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=dataset_color_map[ds], marker='o', linestyle='-',
               linewidth=1, markersize=3, label=dataset_label_map[ds])
        for ds in dataset_names
    ]

    # Reverse the order of legend entries
    legend_handles = legend_handles[::-1]
    legend_labels = [h.get_label() for h in legend_handles]

    fig.legend(
        legend_handles, legend_labels,
        loc='center left', bbox_to_anchor=(0.88, 0.5), borderaxespad=0.
    )

    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='center left', bbox_to_anchor=(0.88, 0.5), borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/' + f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Metrics by model plot saved to 'Figures/{output_prefix}.png'")



def plot_supervised(metrics_by_dataset, output_prefix="metrics_combined"):
    dataset_names = sorted(metrics_by_dataset.keys())
    all_model_names = set()
    all_chunk_sizes = set()
    dataset_info = []

    for dataset_name in metrics_by_dataset:
        dataset_file = 'Datasets/Groupings/' + dataset_name + '.json'
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                size_estimate = len(data)
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            size_estimate = 0

        for model_name, model_data in metrics_by_dataset[dataset_name].items():
            all_model_names.add(model_name)
            for chunk_size in model_data.keys():
                all_chunk_sizes.add(chunk_size)

        dataset_info.append({
            "name": dataset_name,
            "clean_name": re.sub(r'(?<!^)(?=[A-Z])', ' ', dataset_name.replace('.json', '').replace('_', '\n')).replace('Mcmullen','McMullen'),
            "size": size_estimate
        })

    dataset_info.sort(key=lambda x: x["size"], reverse=True)
    dataset_names = [info["name"] for info in dataset_info]
    clean_names = [info["clean_name"] for info in dataset_info]
    dataset_sizes = [info["size"] for info in dataset_info]

    model_names = sorted(all_model_names)
    model_color_map = get_model_colors(model_names)
    print("colors")
    print(model_color_map)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    n_datasets = len(dataset_names)
    if n_datasets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9, 6))
        axes = [axes]
    elif n_datasets == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    elif n_datasets <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
    else:
        n_cols = 3
        n_rows = (n_datasets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols + 1, 3*n_rows))
        axes = axes.flatten()

    for dataset_idx, dataset_name in enumerate(reversed(dataset_names)):
        ax = axes[dataset_idx]
        dataset_data = metrics_by_dataset[dataset_name]
        
        name = list(reversed(clean_names))[dataset_idx]
        size = list(reversed(dataset_sizes))[dataset_idx]
        clean_name = f"{name}\n({size})"


        # Determine chunk sizes used in this dataset
        dataset_chunk_sizes = set()
        for model_data in dataset_data.values():
            dataset_chunk_sizes.update(model_data.keys())
        dataset_chunk_sizes = sorted(dataset_chunk_sizes)

        chunk_size_labels = [str(cs) for cs in dataset_chunk_sizes]
        chunk_size_positions = list(range(len(dataset_chunk_sizes)))

        for i, model_name in enumerate(model_names):
            if model_name not in dataset_data:
                continue

            model_data = dataset_data[model_name]
            multiclass_f1_values = []
            multiclass_f1_errors = []

            for cs in dataset_chunk_sizes:
                if cs in model_data:
                    multiclass_data = model_data[cs].get("multiclass")
                    if multiclass_data and multiclass_data["avg_metrics"] is not None:
                        multiclass_f1_values.append(multiclass_data["avg_metrics"]["macro_f1_score"])
                        multiclass_f1_errors.append(multiclass_data["std_metrics"]["macro_f1_score"])
                    else:
                        multiclass_f1_values.append(None)
                        multiclass_f1_errors.append(None)
                else:
                    multiclass_f1_values.append(None)
                    multiclass_f1_errors.append(None)

            # Filter out None values for plotting
            valid_indices = [i for i, val in enumerate(multiclass_f1_values) if val is not None]
            x_vals = [chunk_size_positions[i] for i in valid_indices]
            y_vals = [multiclass_f1_values[i] for i in valid_indices]
            y_errs = [multiclass_f1_errors[i] for i in valid_indices]

            if y_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                            fmt='o-', label=get_model_shorthand(model_name),
                            linewidth=1, capsize=3, alpha=0.7, color=model_color_map[model_name],
                            markersize=3)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Macro F1 Score')
        ax.set_title(f'{clean_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(chunk_size_positions)
        ax.set_xticklabels(chunk_size_labels)

        # # Add vertical dashed line if chunk sizes include values > 100
        # numeric_chunk_sizes = [int(cs) for cs in dataset_chunk_sizes if str(cs).isdigit()]
        
        # if any(cs > 100 for cs in numeric_chunk_sizes):
        #     greater_than_100 = [cs for cs in numeric_chunk_sizes if cs > 100]
        #     if greater_than_100:
        #         final_index = dataset_chunk_sizes.index(max(greater_than_100))
        #         ax.axvline(x=final_index - 0.5, linestyle='--', color='gray', alpha=0.6)
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.88, 0.5), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('Figures/' + f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    print(f"Combined metrics plot saved to 'Figures/{output_prefix}.png'")

def plot_ari_vs_external(
    metrics_by_dataset,
    *,
    chunk_fixed=1,
    external_summary_path="Artifacts/Group/Summary.json",
    output_prefix="Figure_Supervised",
    external_label=None,
    two_facets=True,
    # Right-facet behavior (what is “max” C for each dataset?)
    max_strategy="dataset_global",         # "dataset_global" (default) or "per_model"
    fallback_if_missing="nearest_lower"    # only for "dataset_global": "nearest_lower" or "skip"
):
    """
    Two-facet scatter of *per (model × dataset)* mean ARI points:
      X-axis: external ARI mean by MODEL (from Artifacts/Group/Summary.json)
      Y-axis: internal ARI mean by (DATASET, MODEL) at a chosen chunk-size rule.

    LEFT facet (if two_facets=True):
        - internal ARI at fixed chunk size == chunk_fixed (e.g., 1)

    RIGHT facet (if two_facets=True):
        - internal ARI at a 'maximum' chunk size per DATASET
          • max_strategy == "dataset_global" (default):
              For each dataset, find the single largest C across ANY model.
              For each model in that dataset, use:
                 - that C if present; else (if fallback_if_missing=="nearest_lower") the largest C <= max;
                 - or skip the point if fallback_if_missing=="skip".
          • max_strategy == "per_model":
              For each (dataset, model), use that model’s largest available C within the dataset.

    INTERNAL ARI source (already computed in your pipeline):
        metrics_by_dataset[dataset][model][C]["multiclass"]["avg_metrics"]["robust_adjusted_rand_score"]
        Optional y-error:
        metrics_by_dataset[dataset][model][C]["multiclass"]["std_metrics"]["robust_adjusted_rand_score"]

    EXTERNAL ARI source:
        Artifacts/Group/Summary.json -> summary.model_performance[model].adjusted_rand_score.{mean, std}

    Coloring/markers:
        - Color encodes MODEL  (via your get_model_colors helper)
        - Marker shape encodes DATASET (stable set of shapes)
        - A small horizontal jitter is applied per dataset so same-model points are readable.

    Output:
        If two_facets:
           Figures/{output_prefix}_ARI_vs_ARI_chunk{chunk_fixed}_vs_max_md.png
        Else (single facet, fixed chunk only):
           Figures/{output_prefix}_ARI_vs_ARI_chunk{chunk_fixed}_md.png
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # ------- Helpers -------
    def _get_entry(md, key):
        """Support numeric or string keys transparently."""
        return md.get(key) if key in md else md.get(str(key))

    def _numeric_chunks(keys):
        out = []
        for k in keys:
            try:
                out.append(int(k))
            except Exception:
                try:
                    out.append(int(str(k)))
                except Exception:
                    pass
        return out

    def _short(s):
        try:
            return get_model_shorthand(s)
        except Exception:
            return s

    def _norm(s):
        return (s or "").lower().replace(" ", "").replace("-", "").replace("_", "")

    # ------- Load EXTERNAL ARI summary -------
    if not os.path.exists(external_summary_path):
        print(f"[plot_ari_vs_external] File not found: {external_summary_path}")
        return

    with open(external_summary_path, "r", encoding="utf-8") as f:
        ext = json.load(f)

    mp = (
        ext.get("summary", {}).get("model_performance")
        or ext.get("model_performance")
        or {}
    )
    external = {}
    for model, stats in mp.items():
        ars = stats.get("adjusted_rand_score", {}) or {}
        m, s = ars.get("mean", None), ars.get("std", None)
        if m is not None:
            external[model] = {"mean": float(m), "std": (None if s is None else float(s))}

    if not external:
        print("[plot_ari_vs_external] No external ARI stats found.")
        return

    # ------- Collect model & dataset universes from internal data -------
    datasets = sorted(list(metrics_by_dataset.keys()))
    models = sorted({m for ds in datasets for m in (metrics_by_dataset.get(ds, {}) or {}).keys()})

    # Colors by MODEL (reuse your helper; fall back if needed)
    try:
        model_color_map = get_model_colors(models)
    except Exception:
        # Fallback palette
        import seaborn as sns
        pal = sns.color_palette("tab20", max(1, len(models)))
        model_color_map = {m: pal[i % len(pal)] for i, m in enumerate(models)}

    # Marker shapes by DATASET (limited set, fine since you have ~6 datasets)
    dataset_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
    ds_marker_map = {ds: dataset_markers[i % len(dataset_markers)] for i, ds in enumerate(datasets)}

    # ------- Build external mapping: internal model name -> external model key -------
    # Match using exact key or normalized shorthand
    def _align_model_key(int_model):
        if int_model in external:
            return int_model
        ems = _norm(_short(int_model))
        for ext_key in external.keys():
            if _norm(_short(ext_key)) == ems:
                return ext_key
        return None

    # ------- Extract INTERNAL ARI points: (dataset, model) for LEFT facet (fixed chunk) -------
    left_points = []  # list of dicts: {ds, model, y_mean, y_std, x_mean, x_std}
    for ds in datasets:
        by_model = metrics_by_dataset.get(ds, {}) or {}
        for m in models:
            by_chunk = by_model.get(m)
            if not by_chunk:
                continue
            entry = _get_entry(by_chunk, chunk_fixed)
            if not entry:
                continue
            mc = entry.get("multiclass") or {}
            avg = mc.get("avg_metrics") or {}
            std = mc.get("std_metrics") or {}
            y_mean = avg.get("robust_adjusted_rand_score", None)
            if y_mean is None:
                continue
            y_std = std.get("robust_adjusted_rand_score", None)

            ext_key = _align_model_key(m)
            if not ext_key:
                continue
            x_mean = external[ext_key]["mean"]
            x_std = external[ext_key]["std"]

            left_points.append({
                "ds": ds, "model": m,
                "y": float(y_mean), "yerr": (None if y_std is None else float(y_std)),
                "x": float(x_mean), "xerr": (None if x_std is None else float(x_std))
            })

    # ------- Extract INTERNAL ARI points: (dataset, model) for RIGHT facet (max chunk rule) -------
    right_points = []
    if two_facets:
        # Pre-compute dataset-global max C if requested
        dataset_global_maxC = {}
        if max_strategy == "dataset_global":
            for ds in datasets:
                by_model = metrics_by_dataset.get(ds, {}) or {}
                all_cs = set()
                for _m, by_chunk in by_model.items():
                    all_cs.update((by_chunk or {}).keys())
                numeric = _numeric_chunks(all_cs)
                dataset_global_maxC[ds] = (max(numeric) if numeric else None)

        for ds in datasets:
            by_model = metrics_by_dataset.get(ds, {}) or {}
            for m, by_chunk in by_model.items():
                if not by_chunk:
                    continue
                chosen_cs = None
                if max_strategy == "per_model":
                    numeric = _numeric_chunks(by_chunk.keys())
                    chosen_cs = (max(numeric) if numeric else None)
                else:  # dataset_global
                    target = dataset_global_maxC.get(ds, None)
                    if target is None:
                        chosen_cs = None
                    else:
                        if _get_entry(by_chunk, target) is not None:
                            chosen_cs = target
                        elif fallback_if_missing == "nearest_lower":
                            numeric = sorted(_numeric_chunks(by_chunk.keys()))
                            lower = [c for c in numeric if c <= target]
                            chosen_cs = (max(lower) if lower else None)
                        else:  # "skip"
                            chosen_cs = None

                if chosen_cs is None:
                    continue

                entry = _get_entry(by_chunk, chosen_cs)
                if not entry:
                    continue
                mc = entry.get("multiclass") or {}
                avg = mc.get("avg_metrics") or {}
                std = mc.get("std_metrics") or {}
                y_mean = avg.get("robust_adjusted_rand_score", None)
                if y_mean is None:
                    continue
                y_std = std.get("robust_adjusted_rand_score", None)

                ext_key = _align_model_key(m)
                if not ext_key:
                    continue
                x_mean = external[ext_key]["mean"]
                x_std = external[ext_key]["std"]

                right_points.append({
                    "ds": ds, "model": m,
                    "y": float(y_mean), "yerr": (None if y_std is None else float(y_std)),
                    "x": float(x_mean), "xerr": (None if x_std is None else float(x_std))
                })

    # ------- If nothing to plot, bail -------
    if not left_points and not right_points:
        print("[plot_ari_vs_external] No points to plot (internal/external mismatch or missing ARI).")
        return

    # ------- Plotting -------
    def _draw(ax, points, title, ylab):
        if not points:
            ax.set_visible(False)
            return

        # Jitter points horizontally by dataset to reveal overlapping same-model x positions
        ds_list = sorted({p["ds"] for p in points})
        ds_to_idx = {d: i for i, d in enumerate(ds_list)}
        nD = max(1, len(ds_list))
        jitter_span = 0.018  # total span across all datasets at a given model
        for p in points:
            ds = p["ds"]
            m  = p["model"]
            x  = p["x"]
            y  = p["y"]
            xerr = p["xerr"]
            yerr = p["yerr"]

            # stable jitter based on dataset index, centered around 0
            offset = ((ds_to_idx[ds] - (nD - 1)/2.0) / max(1, (nD - 1))) * jitter_span if nD > 1 else 0.0
            xj = x + offset
            # keep within plotting range (slightly conservative)
            xj = max(0.0, min(1.05, xj))

            ax.errorbar(
                xj, y,
                xerr=None if xerr is None else [[xerr], [xerr]],
                yerr=None if yerr is None else [[yerr], [yerr]],
                fmt=ds_marker_map.get(ds, 'o'),
                markersize=6,
                markeredgecolor="black",
                markerfacecolor=model_color_map.get(m, "#5D5D5D"),
                ecolor="grey",
                elinewidth=0.9,
                capsize=0,
                alpha=0.95,
                linestyle="none",
                zorder=3
            )

        # Reference 1:1 line and cosmetics
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1, alpha=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xlabel(external_label or "Adjusted Rand (external treatment)")
        ax.set_ylabel(ylab)

        # Build two legends: (1) datasets by marker, (2) models by color
        from matplotlib.lines import Line2D
        # Dataset legend (marker shapes, white fill to highlight shape)
        ds_handles = [
            Line2D([0], [0],
                   marker=ds_marker_map[d], linestyle="None",
                   markerfacecolor="white", markeredgecolor="black", markersize=7,
                   label=d)
            for d in ds_list
        ]
        leg1 = ax.legend(ds_handles, [h.get_label() for h in ds_handles],
                         title="Datasets (marker)", loc="center left",
                         bbox_to_anchor=(1.02, 0.75), frameon=False)
        ax.add_artist(leg1)

        # Model legend (color, consistent marker 'o')
        model_handles = [
            Line2D([0], [0],
                   marker='o', linestyle="None",
                   markerfacecolor=model_color_map.get(m, "#5D5D5D"),
                   markeredgecolor="black", markersize=7,
                   label=_short(m))
            for m in models
        ]
        ax.legend(model_handles, [h.get_label() for h in model_handles],
                  title="Models (color)", loc="center left",
                  bbox_to_anchor=(1.02, 0.25), frameon=False)

    if two_facets:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True, sharey=True, constrained_layout=True)
        ylab_left = f"Adjusted Rand (internal; chunk={chunk_fixed})"
        if max_strategy == "dataset_global":
            ylab_right = "Adjusted Rand (internal; dataset-global max C)"
            title_right = "ARI vs ARI — max chunk per dataset (global)"
        else:
            ylab_right = "Adjusted Rand (internal; per-model max C)"
            title_right = "ARI vs ARI — max chunk per dataset (per-model)"

        title_left = f"ARI vs ARI — chunk={chunk_fixed}"

        _draw(axes[0], left_points,  title_left,  ylab_left)
        _draw(axes[1], right_points, title_right, ylab_right)

        os.makedirs("Figures", exist_ok=True)
        out_path = os.path.join("Figures", f"Figure_ARI_vs_ARI.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Two-facet (model×dataset) ARI vs ARI saved to '{out_path}'")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.0), constrained_layout=True)
        ylab_left = f"Adjusted Rand (internal; chunk={chunk_fixed})"
        _draw(ax, left_points, f"ARI vs ARI — chunk={chunk_fixed}", ylab_left)

        os.makedirs("Figures", exist_ok=True)
        out_path = os.path.join("Figures", f"{output_prefix}_ARI_vs_ARI_chunk{chunk_fixed}_md.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"(model×dataset) ARI vs ARI saved to '{out_path}'")

def plot_dumbbells_by_dataset(
    metrics_by_dataset: dict,
    external_summary_path: str,
    *,
    chunk_key=1,  # int | "max" | Iterable like [1, "max"]
    output_path: str = "Figures/Figure_Dumbbells_ByDataset.png"
):
    """
    Faceted dumbbell plots: for each DATASET (one panel),
    draw lines from Unsupervised (external Group ARI) to Supervised (internal ARI)
    for every MODEL available in that dataset.

    Updates per request:
    - Legend placed OUTSIDE the plot (bottom center).
    - No vertical offset/skew between treatments (same y for a given model).
    - Extra transparency (alpha) to reduce overplotting.
    - Colors are fixed and consistent:
      Unsupervised: grey circle
      Supervised (C=1): green square
      Supervised (C=max): red square

    New in this patch:
    - Facets ordered by dataset size (largest -> smallest), like other plots.
    - Facet titles are "cleaned" and show dataset size, e.g. "Author\nName\n(123)".
    """
    import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, re

    # ------------------------ Helpers ------------------------
    def _get_entry(md, key):
        if key in md: return md[key]
        ks = str(key)
        if ks in md: return md[ks]
        return None

    def _numeric_chunks(keys):
        out = []
        for k in keys:
            try:
                out.append(int(k))
            except Exception:
                try:
                    out.append(int(str(k)))
                except Exception:
                    pass
        return sorted(out)

    def _short(m):
        try:
            return get_model_shorthand(m)
        except:
            return m

    def _as_iterable_chunks(chk):
        # Normalize chunk_key to a list
        try:
            from collections.abc import Iterable
            if isinstance(chk, str) or not isinstance(chk, Iterable):
                return [chk]
            else:
                return list(chk)
        except Exception:
            return [chk]

    # Fixed, consistent colors & markers
    COLOR_UNSUP = "#777777"  # grey
    COLOR_C1 = "#2CA02C"     # green
    COLOR_CMAX = "#D62728"   # red
    MARK_UNSUP = "o"
    MARK_SUP = "s"  # squares for all supervised treatments

    def _treatment_label(ck):
        return f"C={ck}"

    def _treatment_style(treatment: str):
        # Map label -> (color, marker)
        if treatment == "C=1":
            return COLOR_C1, MARK_SUP
        if treatment == "C=max":
            return COLOR_CMAX, MARK_SUP
        # Fallback for any extra treatment
        return "#1f77b4", MARK_SUP  # blue square

    # ------------------------ Load external (Group) ARI by MODEL ------------------------
    if not os.path.exists(external_summary_path):
        print(f"[plot_dumbbells_by_dataset] Missing external summary: {external_summary_path}")
        return

    with open(external_summary_path, "r", encoding="utf-8") as f:
        ext = json.load(f)
    mp = (ext.get("summary", {}) or {}).get("model_performance", {}) or \
         ext.get("model_performance", {}) or {}
    group_ari_mean = {
        m: (stats.get("adjusted_rand_score", {}) or {}).get("mean", None)
        for m, stats in mp.items()
    }

    # ------------------------ Build tidy rows ------------------------
    chunk_keys = _as_iterable_chunks(chunk_key)
    rows = []  # {dataset, model, treatment, group, supervised}

    for ds, by_model in (metrics_by_dataset or {}).items():
        if not by_model:
            continue
        for m, by_chunk in (by_model or {}).items():
            if not by_chunk:
                continue
            grp = group_ari_mean.get(m, None)
            if grp is None:
                continue  # need external ARI to compare
            numeric = _numeric_chunks(by_chunk.keys())
            maxC = numeric[-1] if numeric else None
            for ck in chunk_keys:
                if ck == "max":
                    C = maxC
                else:
                    C = ck
                if C is None:
                    continue
                entry = _get_entry(by_chunk, C)
                if not entry:
                    continue
                mc = entry.get("multiclass") or {}
                avg = mc.get("avg_metrics") or {}
                sup = avg.get("robust_adjusted_rand_score", None)
                if sup is None:
                    continue
                rows.append({
                    "dataset": ds,
                    "model": m,
                    "treatment": _treatment_label(ck),  # e.g., "C=1", "C=max"
                    "group": float(grp),
                    "supervised": float(sup)
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        print("[plot_dumbbells_by_dataset] No data to plot.")
        return

    # ------------------------ NEW: Order facets by dataset size & clean titles ------------------------
    # Build dataset info like in other plots (size + cleaned name).
    dataset_info = []
    for ds in sorted(df["dataset"].unique()):
        dataset_file = f"Datasets/Groupings/{ds}.json"
        size_estimate = 0
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                size_estimate = len(json.load(f))
        except Exception as e:
            print(f"[plot_dumbbells_by_dataset] Error reading {dataset_file}: {e}")
        clean = re.sub(r'(?<!^)(?=[A-Z])', ' ', ds.replace('.json', '').replace('_', '\n')).replace('Mcmullen','McMullen')
        dataset_info.append({"name": ds, "size": size_estimate, "clean": clean})

    # Sort by size (largest first) to match the other figures
    dataset_info.sort(key=lambda d: d["size"], reverse=True)

    # ------------------------ Grid layout ------------------------
    n = len(dataset_info)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.8 * ncols, 3.1 * nrows),  # small height bump to offset 2-line titles
        squeeze=False,
        constrained_layout=True
    )

    # ------------------------ Draw each dataset panel ------------------------
    for idx, info in enumerate(dataset_info):
        ds = info["name"]
        clean = info["clean"]
        size = info["size"]

        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        sub = df[df["dataset"] == ds].copy()

        # Order models to reduce label crossings: mean supervised across treatments
        order = (
            sub.groupby("model")["supervised"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
        sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
        sub = sub.sort_values(["model", "treatment"]).reset_index(drop=True)

        y = np.arange(len(order))
        y_index = {m: i for i, m in enumerate(order)}

        # Draw per-model
        for m in order:
            subm = sub[sub["model"] == m]
            i = y_index[m]
            if subm.empty:
                continue
            grp_val = subm["group"].iloc[0]

            # group marker (unsupervised): grey circle
            ax.scatter(
                grp_val, i,
                s=42, marker=MARK_UNSUP, color=COLOR_UNSUP,
                edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3
            )

            # For each treatment on same row (no vertical offset)
            for _, row in subm.iterrows():
                color_sup, mark_sup = _treatment_style(row["treatment"])
                # line from group to supervised (same y)
                ax.plot([grp_val, row["supervised"]], [i, i],
                        color="gray", lw=1.2, alpha=0.6, zorder=1)
                # supervised marker: colored square
                ax.scatter(
                    row["supervised"], i,
                    s=46, marker=mark_sup, color=color_sup,
                    edgecolor="black", linewidth=0.7, alpha=0.85, zorder=4
                )

        # Y ticks = model short names
        ax.set_yticks(y)
        ax.set_yticklabels([_short(m) for m in order])

        # Cleaned title + size
        ax.set_title(f"{clean}\n({size})")

        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", alpha=0.25)
        ax.axvline(0, ls="--", color="k", alpha=0.15)
        ax.axvline(1, ls="--", color="k", alpha=0.05)

        # Only bottom row gets the x-label
        if r == nrows - 1:
            ax.set_xlabel("Adjusted Rand Index")

        # First column can omit explicit y-label (ticks are enough)
        if c == 0:
            ax.set_ylabel(None)

    # Hide any extra axes (when grid has more cells than datasets)
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].set_visible(False)

    # ------------------------ Legend outside (bottom center) ------------------------
    from matplotlib.lines import Line2D
    present = set(df["treatment"].unique())
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=COLOR_UNSUP,
               markeredgecolor="black", markersize=6, label="Unsupervised"),
    ]
    labels = ["Emergent"]  # keep your preferred display label

    if "C=1" in present:
        handles.append(
            Line2D([0], [0], marker="s", linestyle="None", color=COLOR_C1,
                   markeredgecolor="black", markersize=6, label="Predefined (C=1)")
        )
        labels.append("Predefined (B=1)")

    if "C=max" in present:
        handles.append(
            Line2D([0], [0], marker="s", linestyle="None", color=COLOR_CMAX,
                   markeredgecolor="black", markersize=6, label="Predefined (C=max)")
        )
        labels.append("Predefined (B=max)")

    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.1),
        ncol=max(1, len(handles)), frameon=False
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Dumbbells by dataset saved to '{output_path}'")



def plot_supervised_bar(
    metrics_by_dataset: dict,
    output_prefix: str = "metrics_chunk1_bars",
    *,
    chunk_key=1,                 # filter key for the target chunk size (e.g., 1)
    show_values: bool = False,   # annotate bar tops
    overlay_ari: bool = True,    # overlay robust ARI as dots with error bars
    ari_marker_kwargs: dict | None = None,  # optional marker style overrides for ARI
):
    """
    Grouped (dodged) bar chart for a specific chunk_key (default: 1):
      - X-axis: models (ordered by average Macro F1 across datasets, high -> low)
      - Bar fill (color): dataset (legend shows `clean_name (n)`)
      - Bar height: Macro F1 mean
      - Bar error: Macro F1 std/sem (if available)
      - Optional overlay: Robust Adjusted Rand (AR†) mean as dots + error bars on each bar
        (black dots, error bars without caps)
    """
    import os
    import re
    import json
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # --- Gather dataset sizes and display names --------------------------------
    dataset_info = []
    for dataset_name in metrics_by_dataset:
        dataset_file = f"Datasets/Groupings/{dataset_name}.json"
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                size_estimate = len(json.load(f))
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            size_estimate = 0

        clean = re.sub(r'(?<!^)(?=[A-Z])', ' ',
                       dataset_name.replace('.json', '').replace('_', '\n')).replace('Mcmullen','McMullen')
        dataset_info.append({"name": dataset_name, "clean": clean, "size": size_estimate})

    # Sort for consistent legend ordering (smallest first; flip if preferred)
    dataset_info.sort(key=lambda x: x["size"], reverse=False)
    dataset_names   = [d["name"] for d in dataset_info]
    dataset_display = {d["name"]: f"{d['clean']} ({d['size']})" for d in dataset_info}

    # --- Collect rows for requested chunk_key ----------------------------------
    rows = []       # (dataset, model, macro_f1_mean, macro_f1_err)
    rows_ari = []   # (dataset, model, ari_mean, ari_err)
    models_in_use = set()

    def _get_entry(md: dict, key):
        # support numeric or string keys transparently
        if key in md:
            return md[key]
        ks = str(key)
        if ks in md:
            return md[ks]
        return None

    for ds in dataset_names:
        per_model = metrics_by_dataset.get(ds, {}) or {}
        for model_name, by_chunk in per_model.items():
            entry = _get_entry(by_chunk, chunk_key)
            if not entry:
                continue

            mc  = (entry.get("multiclass") or {})
            avg = (mc.get("avg_metrics") or {})
            std = (mc.get("std_metrics") or {})

            macro = avg.get("macro_f1_score", None)
            if macro is None:
                continue
            macro_err = std.get("macro_f1_score", None)

            rows.append(
                (ds, model_name, float(macro),
                 (None if macro_err is None else float(macro_err)))
            )
            models_in_use.add(model_name)

            # Collect robust ARI if present
            ari_mean = avg.get("robust_adjusted_rand_score", None)
            ari_err  = std.get("robust_adjusted_rand_score", None)
            if ari_mean is not None:
                rows_ari.append(
                    (ds, model_name, float(ari_mean),
                     (None if ari_err is None else float(ari_err)))
                )

    if not rows:
        print(f"[plot_supervised_bar] Nothing to plot for chunk_key == {chunk_key}.")
        return

    # --- Order models by average Macro-F1 (descending) -------------------------
    model_avg_performance = {}
    for model in models_in_use:
        model_scores = [y for ds, m, y, e in rows if m == model]
        model_avg_performance[model] = np.mean(model_scores) if model_scores else 0.0
    model_names = sorted(models_in_use, key=lambda m: model_avg_performance[m], reverse=True)

    # --- Dataset color map ------------------------------------------------------
    dataset_palette   = sns.color_palette("tab20", max(1, len(dataset_names)))
    dataset_color_map = {ds: dataset_palette[i % len(dataset_palette)]
                         for i, ds in enumerate(dataset_names)}

    # --- Optional: display-friendly model names via helper ----------------------
    def _mlabel(m):
        try:
            return get_model_shorthand(m)  # from scripts.helper
        except Exception:
            return m

    # --- Build dense matrices [n_models x n_datasets] for Macro-F1 and ARI -----
    mdl_index = {m: i for i, m in enumerate(model_names)}
    ds_index  = {ds: j for j, ds in enumerate(dataset_names)}
    nM, nD    = len(model_names), len(dataset_names)

    # Macro-F1 means and errors
    Y = np.full((nM, nD), np.nan, dtype=float)
    E = np.full((nM, nD), np.nan, dtype=float)
    for ds, m, y, e in rows:
        i, j = mdl_index[m], ds_index[ds]
        Y[i, j] = y
        if e is not None:
            E[i, j] = e

    # ARI means and errors (may be partially missing)
    R  = np.full((nM, nD), np.nan, dtype=float)
    RE = np.full((nM, nD), np.nan, dtype=float)
    for ds, m, rmean, rerr in rows_ari:
        i, j = mdl_index[m], ds_index[ds]
        R[i, j] = rmean
        if rerr is not None:
            RE[i, j] = rerr

    # --- Plot grouped bars ------------------------------------------------------
    os.makedirs("Figures", exist_ok=True)
    fig_w = max(8.0, 0.8 * nM + 3)
    fig_h = 5.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    x = np.arange(nM)
    # keep total group width <= 0.9
    total_group_w = min(0.9, 0.15 * nD + 0.3)
    bar_w = total_group_w / max(nD, 1)
    # center dataset bars around each model tick
    offsets = (np.arange(nD) - (nD - 1) / 2.0) * bar_w

    for j, ds in enumerate(dataset_names):
        y = Y[:, j]
        e = E[:, j]
        xpos = x + offsets[j]

        # Bars: Macro-F1
        ax.bar(
            xpos, y, width=bar_w,
            color=dataset_color_map.get(ds, "#5D5D5D"),
            edgecolor="black", linewidth=0.3,
            label=dataset_display[ds], alpha=0.95
        )

        # Error bars for Macro-F1 (with caps)
        mask = np.isfinite(y) & np.isfinite(e)
        if mask.any():
            ax.errorbar(
                xpos[mask], y[mask], yerr=e[mask],
                fmt="none", ecolor='grey',
                elinewidth=1.2, capsize=3, alpha=0.9  # <-- keep caps for F1
            )

        # Optional numeric labels above bars
        if show_values:
            for xi, yi in zip(xpos, y):
                if np.isfinite(yi):
                    ax.text(xi, yi + 0.02, f"{yi:.2f}",
                            ha="center", va="bottom", fontsize=8)

        # --- Overlay ARI as dots + error bars (no caps, black markers) ---------
        if overlay_ari:
            r  = R[:, j]
            re = RE[:, j]
            mask_r = np.isfinite(r)
            if mask_r.any():
                # Default style: solid black dots, no caps on error bars
                default_ari_kwargs = dict(
                    fmt='o',
                    color='k',               # line/edge color
                    markerfacecolor='k',     # solid black fill
                    markeredgecolor='k',
                    markersize=2,
                    elinewidth=1.2,
                    capsize=0,               # <-- no 'T' caps for ARI
                    zorder=5,
                    alpha=0.85
                )
                if ari_marker_kwargs:
                    default_ari_kwargs.update(ari_marker_kwargs)

                # yerr: pass only if any finite; else None
                yerr_vals = re[mask_r] if (isinstance(re, np.ndarray) and np.isfinite(re[mask_r]).any()) else None

                ax.errorbar(
                    xpos[mask_r], r[mask_r],
                    yerr=yerr_vals,
                    **default_ari_kwargs
                )

    # --- Axes cosmetics ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels([_mlabel(m) for m in model_names], rotation=30, ha="right")
    ax.set_ylabel("Macro F1 (bars) / Robust Adjusted Rand (dots)")
    # ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # --- Single combined legend: datasets + ARI key ----------------------------
    # Build proxy handles for datasets to avoid duplicate bar entries
    dataset_handles = [
        Patch(facecolor=dataset_color_map[ds], edgecolor="black",
              label=dataset_display[ds])
        for ds in dataset_names
    ]
    ari_handle = Line2D(
        [0], [0],
        marker='o', linestyle='None',
        color='k', markerfacecolor='k', markeredgecolor='k',
        markersize=5,
        label='Adjusted Rand (mean ± err)'  # shown in same legend
    )
    # Combine (datasets first, then ARI annotation)
    legend_handles = dataset_handles + [ari_handle]

    leg = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Dataset (n)"
    )
    if leg and leg.get_title():
        leg.get_title().set_fontsize(10)

    # Save
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # space for legend at right
    out_path = os.path.join("Figures", f"{output_prefix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Chunk={chunk_key} grouped bar plot saved to '{out_path}'")
    # # ===============================
    # # NEW: Scatter — Macro F1 (x) vs Robust ARI† (y)
    # # ===============================
    # # Only draw if we actually have any ARI values
    # if np.isfinite(R).any():
    #     # Build a consistent marker map for models
    #     model_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
    #     marker_map = {m: model_markers[i % len(model_markers)] for i, m in enumerate(model_names)}

    #     fig_sc, ax_sc = plt.subplots(1, 1, figsize=(8,4),constrained_layout = True)

    #     # Plot each (model, dataset) point with dataset color and model-shaped marker
    #     for i, m in enumerate(model_names):
    #         for j, ds in enumerate(dataset_names):
    #             x = Y[i, j]   # Macro-F1 mean
    #             y = R[i, j]   # ARI† mean
    #             if not (np.isfinite(x) and np.isfinite(y)):
    #                 continue

    #             xerr = E[i, j] if (j < E.shape[1] and np.isfinite(E[i, j])) else None
    #             yerr = RE[i, j] if (j < RE.shape[1] and np.isfinite(RE[i, j])) else None

    #             ax_sc.errorbar(
    #                 x, y,
    #                 xerr=None if xerr is None else [[xerr], [xerr]],  # symmetric
    #                 yerr=None if yerr is None else [[yerr], [yerr]],
    #                 fmt=marker_map[m],
    #                 markersize=5,
    #                 markerfacecolor=dataset_color_map.get(ds, "#5D5D5D"),
    #                 markeredgecolor="black",
    #                 ecolor="grey",
    #                 elinewidth=0.8,
    #                 capsize=1,
    #                 alpha=0.9,
    #                 linestyle="none",
    #                 zorder=3
    #             )

    #     # 1:1 reference line
    #     ax_sc.plot([0, 1], [0, 1], linestyle="--", color="k", alpha=0.6, linewidth=1)

    #     # Axes cosmetics
    #     ax_sc.set_xlim(0, 1.05)
    #     ax_sc.set_ylim(0, 1.05)
    #     ax_sc.set_xlabel("Macro F1 (mean ± sd)")
    #     ax_sc.set_ylabel("Robust Adjusted Rand, ARI† (mean ± sd)")
    #     ax_sc.grid(True, alpha=0.25)
    #     ax_sc.set_axisbelow(True)
    #     ax_sc.set_title(f"Macro F1 vs ARI† (chunk={chunk_key})")

    #     # Legends: (1) datasets by color, (2) models by marker shape
    #     # Dataset legend
    #     dataset_handles = [
    #         Patch(facecolor=dataset_color_map[ds], edgecolor="black", label=dataset_display[ds])
    #         for ds in dataset_names
    #     ]
    #     leg1 = ax_sc.legend(
    #         handles=dataset_handles,
    #         title="Dataset (n)",
    #         loc="center left",
    #         bbox_to_anchor=(1.0, 0.75),
    #         frameon=False
    #     )
    #     if leg1 and leg1.get_title():
    #         leg1.get_title().set_fontsize(10)
    #     ax_sc.add_artist(leg1)

    #     # Model legend (use black outlines; fill white to emphasize shape)
    #     model_handles = [
    #         Line2D([0], [0],
    #                marker=marker_map[m], linestyle="None",
    #                markerfacecolor="white", markeredgecolor="black",
    #                markersize=7, label=(get_model_shorthand(m) if callable(get_model_shorthand) else m))
    #         for m in model_names
    #     ]
    #     leg2 = ax_sc.legend(
    #         handles=model_handles,
    #         title="Models (marker)",
    #         loc="center left",
    #         bbox_to_anchor=(1.0, 0.25),
    #         frameon=False
    #     )
    #     if leg2 and leg2.get_title():
    #         leg2.get_title().set_fontsize(10)

    #     # Save the scatter figure alongside the bar chart
    #     scatter_out = os.path.join("Figures", f"{output_prefix}_F1_vs_ARI.png")
    #     plt.tight_layout(rect=[0,0,0.75,1])  # make room for legends on the right
    #     fig_sc.savefig(scatter_out, dpi=300, bbox_inches="tight")
    #     print(f"F1 vs ARI scatter saved to '{scatter_out}'")
    # else:
    #     print("[plot_supervised_bar] No finite ARI values found; skipping F1–ARI scatter.")


import math
import ast

def _is_missing(x):
    # Treat None, NaN, and empty/whitespace strings as missing
    return (
        x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and x.strip() == "")
        or isinstance(x, (list, tuple))

    )

def sanitize_content(content, *, label_field="group", missing_label="missing"):
    """
    Sanitize predictions in `content` by mapping only the label field to `missing`
    when absent. Does NOT convert dicts to strings.
    Handles common shapes:
      - list of dicts: [{..., label_field: <label>}, ...]
      - list of labels: [<label>, ...]
      - dict mapping id -> label OR id -> dict-with-label
      - single dict or single label
    Also recovers from *stringified dicts* (caused by earlier bug) using ast.literal_eval.
    """
    def fix_item(item):
        # Recover if item is a stringified dict (from previous run)
        if isinstance(item, str) and item.strip().startswith("{") and f"'{label_field}'" in item:
            try:
                item = ast.literal_eval(item)  # safe parse of Python literal
            except Exception:
                # Not a parsable dict; treat it as a bare label below
                pass

        if isinstance(item, dict):
            val = item.get(label_field)
            if _is_missing(val):
                item[label_field] = missing_label
            return item
        else:
            # Bare label
            return missing_label if _is_missing(item) else item

    # Shape handlers
    if isinstance(content, list):
        return [fix_item(it) for it in content]
    elif isinstance(content, dict):
        out = {}
        for k, v in content.items():
            if isinstance(v, dict):
                out[k] = fix_item(v)
            else:
                out[k] = missing_label if _is_missing(v) else v
        return out
    else:
        # Single value (dict or label)
        return fix_item(content)


def _get_entry(md, key):
    return md.get(key) if key in md else md.get(str(key))


def run_agreement_analysis(results_dir="results"):
    """
    Run the agreement analysis pipeline.

    Args:
        results_dir: Directory containing result files

    Returns:
        tuple: (analysis_results, chunk_effect_results, metrics_by_chunk_size)
    """
    # Read files
    result_files = read_result_files(results_dir)

    # Analyze files
    analysis_results = {}
    # folder_path = "Datasets"
    # file_list = [os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files]

    for filename, result_list in result_files.items():
        for result_obj in result_list:
            # Ensure result_obj has 'metadata' and 'content'
            if not isinstance(result_obj, dict) or 'metadata' not in result_obj or 'content' not in result_obj:
                print(f"Skipping malformed result file: {filename}")
                continue

            metadata = result_obj['metadata']
            content = result_obj['content']
            
            # ✅ Only touch the 'group' field (or set label_field to whatever holds the prediction)
            content = sanitize_content(content, label_field="group", missing_label="missing")

            # If you want later code to see the sanitized values, write back *only the field*:
            result_obj['content'] = content

            dataset = metadata.get('dataset')
            dataset_file = 'Datasets/Groupings/' + dataset + '.json'
            model = metadata.get('model')
            C = int(metadata.get('C', 0))
            # N = int(metadata.get('N', 100))
            M = int(metadata.get('M', 0))

            # if N < 100:
            #     N = 100

            
            human_classifications = load_human_classifications(dataset_file)
            try:
                human_analysis = analyze_human_agreement(content, human_classifications, dataset_file)
            except:
                breakpoint()
            save_key = f"{dataset}_{model}_{C}_{M}"


            analysis_results[save_key] = {
                "file": filename,
                "C": C,
                "M": M,
                "model": model,
                "dataset": dataset,
                "analysis": human_analysis
            }
    # ---- NEW: Fit hierarchical model testing batch-size effects ----
    
    # save_analysis_results(analysis_results)
    # generate_qualitative_summary(analysis_results)
    # generate_accuracy_heatmap('Artifacts/Classify/Results-Qualitative.json')

        # 2) Build fit results (fits + threshold are drawn if coefs present)
    

    # Calculate and plot metrics by chunk size
    metrics_by_chunk_size = calculate_metrics_by_chunk_size(analysis_results)
    
    summary_df = run_model_selection_with_bootstrap_and_plot(
    analysis_results=analysis_results,                  # ignored in replot-only mode
    metrics_by_dataset=metrics_by_chunk_size,
    out_dir="Artifacts/ModelSelection",   # where the CSV lives
    fig_prefix="Figure_Batch_Modelled",
    replot_only=True,                     # <- NEW
    summary_csv=None
)
    model_fit_results = build_model_fit_results_from_summary(summary_df)








#  3) Plot – the updated plotter will draw model fit pre/post-T segments
    
    # plot_metrics_by_model_with_fits(
    #     metrics_by_chunk_size,
    #     model_fit_results=model_fit_results,
    #     output_prefix="decline_after_one",
    #     alpha_signif=0.05,     # for pre‑T significance
    #     prefer_break_p=True,
    #     alpha_confirm=0.05,    # NEW
    #     alpha_explor=0.10      # NEW
    # )



    # plot_metrics_by_chunk_size(metrics_by_chunk_size, "metrics_by_chunk_size")
    plot_supervised_bar(metrics_by_chunk_size, "Figure_Supervised")
    plot_metrics_by_model(metrics_by_chunk_size, "Figure_Supervised_Model")
    # New: ARI (this analysis) vs ARI (external treatment) by model
    # try:
    #     plot_ari_vs_external(
    #         metrics_by_chunk_size,
    #         chunk_fixed=1,  # change if you want a different C
    #         external_summary_path="Artifacts/Group/Summary.json",
    #         output_prefix="Figure_Supervised",
    #         external_label="Adjusted Rand (Group treatment)",  # optional, pretty axis text
    #         # internal_label="Adjusted Rand (Supervised; chunk=1)"
    #     )
    # except Exception as e:
    #     print(f"[plot_ari_vs_external] Skipped: {e}")


    

    plot_dumbbells_by_dataset(metrics_by_chunk_size, "Artifacts/Group/Summary.json",
                          chunk_key=[1, "max"])

                        


    # print("#"*50)
    # print('Analysis Results')
    # print(analysis_results)
    # print("#"*50)

    # print("Metrics by Batch Size")
    # print(metrics_by_chunk_size)


    return analysis_results, metrics_by_chunk_size

if __name__ == "__main__":
    # Run analysis on results_N20 directory
    run_agreement_analysis(results_dir="Artifacts/Classify/results")