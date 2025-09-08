import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os
from scripts.helper import get_model_shorthand,get_model_colors
from collections import Counter
import re
# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_summary_data(file_path):
    """Load the Group-Summary.json data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def plot_group_sizes(directory, save_dir):
    plt.rcdefaults()
    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    # Dictionary to hold group size frequencies per file and total size
    file_data = {}

    # Set to collect all unique group sizes
    all_group_sizes = set()

    for json_file in json_files:
        file_path = os.path.join(directory, json_file)

        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Count the number of taxon per group
        group_counts = Counter(data.values())

        # Count the frequency of group sizes
        size_frequency = Counter(group_counts.values())

        # Calculate the total number of groups for this file
        total_groups = sum(size_frequency.values())

        # Store the frequency and total size for this file
        file_data[json_file] = {'freq': size_frequency, 'total': total_groups}

        # Update the set of all group sizes
        all_group_sizes.update(size_frequency.keys())

    # Sort all group sizes for the x-axis
    sorted_group_sizes = sorted(all_group_sizes)

    # Sort the files by total number of groups, descending
    sorted_files = sorted(file_data.keys(), key=lambda f: file_data[f]['total'], reverse=True)

    # Prepare data for stacked histogram
    histogram_data = []
    cleaned_labels = []

    for file in sorted_files:
        freq = file_data[file]['freq']
        histogram_data.append([freq.get(size, 0) for size in sorted_group_sizes])
        cleaned_labels.append(file.replace('_', ' - ').replace('.json', '')).replace('Mcmullen','McMullen')

    # Use a nicer color scheme
    colors = cm.get_cmap('Accent', len(sorted_files)).colors

    # Define bins - you can adjust these as needed
    bin_edges = [1, 2, 3, 5, 9, 17, float('inf')]
    bin_labels = ['1', '2', '3-4', '5-8', '9-16', '>16']

    # Function to assign group size to bin
    def get_bin_index(group_size):
        for i, edge in enumerate(bin_edges[1:]):
            if group_size < edge:
                return i
        return len(bin_labels) - 1

    # Reprocess data with binning
    binned_file_data = {}
    for file in sorted_files:
        freq = file_data[file]['freq']
        binned_freq = [0] * len(bin_labels)
        
        for group_size, frequency in freq.items():
            bin_idx = get_bin_index(group_size)
            binned_freq[bin_idx] += frequency
        
        binned_file_data[file] = binned_freq

    # Prepare data for plotting
    histogram_data = [binned_file_data[file] for file in sorted_files]

    # Plot dodged histogram
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
    ax.set_facecolor('white')

    # Create x positions for bins
    x_indices = np.arange(len(bin_labels))

    # Calculate bar width and positions for dodging
    n_files = len(sorted_files)
    bar_width = 0.8 / n_files

    for i, file_data_binned in enumerate(histogram_data):
        # Calculate the x position for this group of bars
        x_offset = (i - n_files/2 + 0.5) * bar_width
        x_pos = x_indices + x_offset
        
        ax.bar(x_pos, file_data_binned, width=bar_width, label=cleaned_labels[i], color=colors[i])

    # Set the color of the axes, labels, and ticks to dark grey
    dark_grey = '#333333'
    ax.set_xlabel("Number of Taxa in Group", color=dark_grey)
    ax.set_ylabel("Number of Groups", color=dark_grey)

    # Change the color of the axis lines (spines)
    ax.spines['left'].set_color(dark_grey)
    ax.spines['bottom'].set_color(dark_grey)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Change the color of the tick labels
    ax.tick_params(axis='x', colors=dark_grey)
    ax.tick_params(axis='y', colors=dark_grey)

    # Set x-ticks to show the bin labels
    ax.set_xticks(x_indices)
    ax.set_xticklabels(bin_labels)

    ax.legend()
    fig.tight_layout()
    ax.grid(False)
    fig.savefig(os.path.join(save_dir, 'Figure_ClusterBalance.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(save_dir, 'Figure_ClusterBalance.pdf'), dpi=300, bbox_inches='tight', facecolor='white')



def extract_ari_data(summary_data):
    """Extract ARI data for analysis and plotting"""
    model_data = []
    dataset_data = []
    replicate_data = []

    # Extract model performance data
    for model, perf in summary_data['summary']['model_performance'].items():
        ari_info = perf.get('adjusted_rand_score', {})
        if ari_info.get('mean') is not None:
            model_data.append({
                'model': model,
                'mean_ari': ari_info['mean'],
                'std_ari': ari_info.get('std', 0),
                'sem_ari': (ari_info.get('std', 0) / np.sqrt(ari_info['count'])) if ari_info.get('std') is not None and ari_info['count'] > 0 else 0,
                'count': ari_info['count'],
                'min_ari': ari_info['min'],
                'max_ari': ari_info['max'],
                'datasets_analyzed': perf['datasets_analyzed'],
                'total_replicates': perf['total_replicates']
            })

    # Extract dataset summary data
    for dataset, summary in summary_data['summary']['dataset_summary'].items():
        ari_info = summary.get('adjusted_rand_score', {})
        if ari_info.get('mean') is not None:
            dataset_data.append({
                'dataset': dataset,
                'dataset_size': summary['dataset_size'],
                'reference_groups': summary['reference_groups_count'],
                'mean_ari': ari_info['mean'],
                'sem_ari': np.sqrt(ari_info['mean'] * (1 - ari_info['mean']) / ari_info['count']) if ari_info['count'] > 0 else 0,
                'count': ari_info['count'],
                'min_ari': ari_info['min'],
                'max_ari': ari_info['max'],
                'models_analyzed': summary['models_analyzed'],
                'total_replicates': summary['total_replicates']
            })

    # Extract individual replicate data for detailed analysis
    for dataset_name, dataset_info in summary_data['detailed_results'].items():
        dataset_size = dataset_info['reference_info']['total_taxon']
        reference_groups = dataset_info['reference_info']['total_groups']

        for model_name, model_info in dataset_info['models'].items():
            for replicate in model_info['replicates']:
                ari_info = replicate.get('adjusted_rand_score')
                if ari_info and ari_info.get('score') is not None:
                    replicate_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'dataset_size': dataset_size,
                        'reference_groups': reference_groups,
                        'ari_score': ari_info['score'],
                        'replicate_id': replicate['replicate_id']
                    })
    return pd.DataFrame(model_data), pd.DataFrame(dataset_data), pd.DataFrame(replicate_data)

def extract_ami_data(summary_data):
    """
    Extract AMI data for analysis and plotting.
    Mirrors extract_ari_data, but searches for AMI in multiple likely keys to be robust.
    Returns (model_df, dataset_df, replicate_df) with 'ami_score' in replicate_df.
    """
    AMI_KEYS = [
        'adjusted_mutual_info_score',
        'adjusted_mutual_info',
        'adjusted_mutual_information_score',
        'adjusted_mutual_information',
        'ami'
    ]

    def _get_metric_info(container):
        # Returns the metric dict if found; otherwise empty dict
        for k in AMI_KEYS:
            val = container.get(k)
            if val is not None:
                return val
        return {}

    model_data, dataset_data, replicate_data = [], [], []

    # --- Model performance ---
    for model, perf in summary_data['summary']['model_performance'].items():
        ami_info = _get_metric_info(perf)
        # Accept either dict-like (mean/std/… present) or a direct scalar 'score'
        if isinstance(ami_info, dict) and ami_info.get('mean') is not None:
            count = ami_info.get('count', 0) or 0
            std = ami_info.get('std', 0) or 0
            sem = (std / np.sqrt(count)) if std is not None and count > 0 else 0
            model_data.append({
                'model': model,
                'mean_ami': ami_info['mean'],
                'std_ami': std,
                'sem_ami': sem,
                'count': count,
                'min_ami': ami_info.get('min', np.nan),
                'max_ami': ami_info.get('max', np.nan),
                'datasets_analyzed': perf.get('datasets_analyzed', np.nan),
                'total_replicates': perf.get('total_replicates', np.nan),
            })

    # --- Dataset summary ---
    for dataset, summary in summary_data['summary']['dataset_summary'].items():
        ami_info = _get_metric_info(summary)
        if isinstance(ami_info, dict) and ami_info.get('mean') is not None:
            count = ami_info.get('count', 0) or 0
            mean = ami_info['mean']
            # Same SEM convention as ARI block in your file
            sem = np.sqrt(mean * (1 - mean) / count) if count > 0 else 0
            dataset_data.append({
                'dataset': dataset,
                'dataset_size': summary['dataset_size'],
                'reference_groups': summary['reference_groups_count'],
                'mean_ami': mean,
                'sem_ami': sem,
                'count': count,
                'min_ami': ami_info.get('min', np.nan),
                'max_ami': ami_info.get('max', np.nan),
                'models_analyzed': summary.get('models_analyzed', np.nan),
                'total_replicates': summary.get('total_replicates', np.nan),
            })

    # --- Individual replicates ---
    for dataset_name, dataset_info in summary_data['detailed_results'].items():
        dataset_size = dataset_info['reference_info']['total_taxon']
        reference_groups = dataset_info['reference_info']['total_groups']
        for model_name, model_info in dataset_info['models'].items():
            for replicate in model_info['replicates']:
                # Try the candidate keys on the replicate itself
                metric_entry = None
                for k in AMI_KEYS:
                    if k in replicate and isinstance(replicate[k], dict):
                        metric_entry = replicate[k]
                        break
                if metric_entry and metric_entry.get('score') is not None:
                    replicate_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'dataset_size': dataset_size,
                        'reference_groups': reference_groups,
                        'ami_score': metric_entry['score'],
                        'replicate_id': replicate.get('replicate_id')
                    })

    return pd.DataFrame(model_data), pd.DataFrame(dataset_data), pd.DataFrame(replicate_data)


def plot_dataset_size_effect_ami(dataset_df, replicate_df, save_dir):
    """
    Plot Dataset Size vs AMI Performance with model mean lines and labels.
    Layout/colors/markers match plot_dataset_size_effect (ARI version).
    Saves: Figure_Unsupervised_AMI.png / .pdf
    """
    fig, ax1 = plt.subplots(figsize=(7, 4))
    sns.set_style("white")
    ax1.set_facecolor("white")
    fig.patch.set_facecolor("white")
    sns.despine(ax=ax1)

    # Prepare data for boxplot — group by dataset and sort by size
    dataset_info = []
    for dataset in replicate_df['dataset'].unique():
        ddf = replicate_df[replicate_df['dataset'] == dataset]
        dataset_size = ddf['dataset_size'].iloc[0]
        ami_values = ddf['ami_score'].dropna().values
        if len(ami_values) > 0:
            dataset_info.append({'name': dataset, 'size': dataset_size, 'data': ami_values})

    dataset_info.sort(key=lambda x: x['size'])
    datasets = [d['name'] for d in dataset_info]
    dataset_sizes = [d['size'] for d in dataset_info]
    boxplot_data = [d['data'] for d in dataset_info]
    positions = list(range(len(datasets)))
    for pos in positions:
        ax1.axvline(pos + 0.5, color='white', linewidth=1, zorder=1)

    # Create boxplot
    bp = ax1.boxplot(boxplot_data, positions=positions, widths=0.4, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('grey')
        patch.set_alpha(0.4)
    for whisker in bp['whiskers']:
        whisker.set_color('grey'); whisker.set_alpha(0.4)
    for cap in bp['caps']:
        cap.set_color('grey'); cap.set_alpha(0.4)
    for median in bp['medians']:
        median.set_color('grey'); median.set_alpha(0.4)

    # Markers/colors identical to ARI plot
    models = sorted(replicate_df['model'].unique())
    markers = ['o', '^', 's','D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    model_markers = {m: markers[i % len(markers)] for i, m in enumerate(models)}
    model_color_map = get_model_colors(models)

    # Individual replicate points with beeswarm jitter (re-use your helper)
    for i, dataset in enumerate(datasets):
        ddf = replicate_df[replicate_df['dataset'] == dataset]
        for model in ddf['model'].unique():
            mdf = ddf[ddf['model'] == model]
            x_coords = beeswarm_jitter(mdf['ami_score'], center=i)
            ax1.scatter(
                x_coords, mdf['ami_score'],
                alpha=0.8, s=10, color=model_color_map[model],
                marker=model_markers[model], zorder=3
            )

    # Model means + labels column (mirrors ARI)
    model_means = []
    for model in models:
        mdf = replicate_df[replicate_df['model'] == model]
        model_means.append((model, mdf['ami_score'].mean()))
    model_means.sort(key=lambda x: x[1])

    model_min = min(model_means, key=lambda x: x[1])
    model_max = max(model_means, key=lambda x: x[1])

    mean_x_pos = len(datasets) + 0.05
    column_width = 0.6
    col_left = mean_x_pos - column_width / 2
    col_right = mean_x_pos + column_width / 2
    fixed_label_x = col_right + 0.25
    label_spacing = 0.08

    sorted_models = sorted(model_means, key=lambda x: x[1], reverse=True)
    label_positions = []
    for i, (model, mean_ami) in enumerate(sorted_models):
        y = mean_ami
        ax1.hlines(y, col_left, col_right, linestyles="--", color="grey", linewidth=1, zorder=2)
        if i == 0:
            final_label_y = y + label_spacing
        else:
            final_label_y = label_positions[-1][1] - label_spacing
        label_positions.append((fixed_label_x, final_label_y))

        # connector
        ax1.plot([col_right, fixed_label_x], [y, final_label_y], color='black', linewidth=0.5, zorder=3)

        # text + marker
        label_text = "     " + get_model_shorthand(model)
        ax1.text(
            fixed_label_x, final_label_y, label_text,
            fontsize=8, va='center', zorder=5, ha='left',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round,pad=0.2')
        )
        ax1.scatter(
            fixed_label_x + 0.12, final_label_y,
            alpha=0.9, s=40, color=model_color_map[model],
            marker=model_markers[model], zorder=6
        )

    # Axis formatting (match ARI)
    import re as _re  # local import to avoid name collision
    dataset_labels = [
        f"{_re.sub(r'(?<!^)(?=[A-Z])', '\\n', ds.replace('.json','').replace('_',''))}\n(n={n})"
        for ds, n in zip(datasets, dataset_sizes)
    ]
    dataset_labels.append("Means")

    ax1.set_xticks(positions + [mean_x_pos])
    ax1.set_xticklabels(dataset_labels, fontsize=9)
    ax1.set_ylabel('AMI Score')
    ax1.grid(False)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)  # match ARI padding, even though AMI ≥ 0

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Figure_Unsupervised_AMI.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(save_dir, 'Figure_Unsupervised_AMI.pdf'),
                dpi=300, bbox_inches='tight', facecolor='white')

# Beeswarm-like jittering function

def beeswarm_jitter(values, center, width=0.3, min_dist=0.1, max_attempts=15000):
    sorted_vals = sorted(values)
    jittered_x = []
    occupied = []

    for val in sorted_vals:
        attempt = 0
        x = center
        while any(abs(x - ox) < min_dist for ox in occupied) and attempt < max_attempts:
            x = center + np.random.uniform(-width, width)
            attempt += 1
        occupied.append(x)
        jittered_x.append(x)
    return jittered_x




def plot_dataset_size_effect(dataset_df, replicate_df, save_dir):


    """Plot Dataset Size vs ARI Performance with Model Mean Points and Labels"""
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    sns.set_style("white")
    ax1.set_facecolor("white")
    fig.patch.set_facecolor("white")
    sns.despine(ax=ax1)

    # Prepare data for boxplot - group by dataset and sort by size
    dataset_info = []
    for dataset in replicate_df['dataset'].unique():
        dataset_data = replicate_df[replicate_df['dataset'] == dataset]
        dataset_size = dataset_data['dataset_size'].iloc[0]
        
        # Filter out NaN values from ARI scores
        ari_values = dataset_data['ari_score'].dropna().values
        if len(ari_values) > 0:
            dataset_info.append({
                'name': dataset,
                'size': dataset_size,
                'data': ari_values
            })

    
    # Sort by dataset size
    dataset_info.sort(key=lambda x: x['size'])
    
    # Extract sorted data
    datasets = [info['name'] for info in dataset_info]
    dataset_sizes = [info['size'] for info in dataset_info]
    boxplot_data = [info['data'] for info in dataset_info]
    
    # Create categorical positions (0, 1, 2, ...)
    positions = list(range(len(datasets)))
    
    for pos in positions:
        ax1.axvline(pos + 0.5, color='white', linewidth=1, zorder=1)

    # Add horizontal bands for ARI score interpretation
    interpretation_bands = [
        (0.0, 0.65, '#ffcccc', 'Poor'),
        (0.65, 0.8, '#ffffcc', 'Moderate'),
        (0.8, 0.9, '#e6ffcc', 'Good'),
        (0.9, 1.0, '#ccffcc', 'Excellent')
    ]
        # Add model mean points and labels
    mean_x_pos = len(datasets) + 0.05
    # for y_min, y_max, color, label in interpretation_bands:
    #     ax1.axhspan(y_min, y_max, xmin=0, xmax=(mean_x_pos + 0.5) / (len(datasets) + 1), alpha=0.8, color=color, zorder=0)


    # Create boxplot
    bp = ax1.boxplot(boxplot_data, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=False)

    # Set alpha for boxes
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('grey')
        patch.set_alpha(0.4)

    # Set alpha for whiskers
    for whisker in bp['whiskers']:
        whisker.set_color('grey')
        whisker.set_alpha(0.4)

    # Set alpha for caps
    for cap in bp['caps']:
        cap.set_color('grey')
        cap.set_alpha(0.4)

    # Set alpha for medians
    for median in bp['medians']:
        median.set_color('grey')
        median.set_alpha(0.4)

    # Optionally set alpha for fliers if you decide to show them
    # for flier in bp['fliers']:
    #     flier.set_alpha(0.4)

    
    # Define marker styles for different models
    models = sorted(replicate_df['model'].unique())
    markers = ['o',  '^', 's','D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    print(model_markers)
    model_color_map = get_model_colors(models)
    print('colors')
    print(model_color_map)
    # # Add individual replicate points
    # def beeswarm_jitter(values, center, width=0.4):
    #     return np.random.normal(loc=center, scale=width/4, size=len(values))

    for i, dataset in enumerate(datasets):
        dataset_data = replicate_df[replicate_df['dataset'] == dataset]
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model]
            x_coords = beeswarm_jitter(model_data['ari_score'], center=i)
            ax1.scatter(x_coords, model_data['ari_score'],
                        alpha=0.8, s=10, color=model_color_map[model],
                        # marker='o', 
                        marker=model_markers[model],
                        zorder=3
                        # edgecolors='black', linewidth=0.1
                        )


        
    
    # Calculate mean ARI scores for each model
    model_means = []
    # models.remove('ollama_hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL')
    for model in models:
        model_data = replicate_df[replicate_df['model'] == model]
        mean_ari = model_data['ari_score'].mean()
        model_means.append((model, mean_ari))

    # Sort by mean ARI to help with label placement
    model_means.sort(key=lambda x: x[1])
    # breakpoint()
    model_min = min(model_means, key=lambda x: x[1])

    model_max = max(model_means, key=lambda x: x[1])
    # Generate evenly spaced label y-positions
    
    label_y_positions = np.linspace(model_min[1], model_max[1], len(model_means))

    

    label_positions = []
    line_segments = []

    # Column bounds for the Model Means column
    column_width = 0.6
    col_left = mean_x_pos - column_width / 2
    col_right = mean_x_pos + column_width / 2


    # Fixed x position for all labels (shifted right for readability)
    fixed_label_x = col_right + 0.25   
    label_spacing = 0.08  
    # Sort models by mean_ari (highest → lowest)
    sorted_models = sorted(model_means, key=lambda x: x[1], reverse=True)
    
    for i, (model, mean_ari) in enumerate(sorted_models):
        y = mean_ari  

        # --- Horizontal dashed line across the entire Model Means column ---
        ax1.hlines(
            y, col_left, col_right,
            linestyles="--", color="grey", linewidth=1, zorder=2
        )


        # --- Label positioning ---
        if i == 0:
            # Highest ARI label above
            final_label_y = y + label_spacing
        else:
            # Cascade labels downward
            final_label_y = label_positions[-1][1] - label_spacing

        label_positions.append((fixed_label_x, final_label_y))

        # --- Connector line: from right edge of dashed line → label ---
        line_segments.append(((col_right, y), (fixed_label_x, final_label_y)))
        ax1.plot(
            [col_right, fixed_label_x], [y, final_label_y],
            color='black', linewidth=0.5, zorder=3
        )

        # --- Add label text, with whitespace prefix ---
        label_text = "        " + get_model_shorthand(model)  # prepend whitespace
        ax1.text(
            fixed_label_x, final_label_y, label_text,
            fontsize=8, verticalalignment='center', zorder=5,
            ha='left',  # anchor left
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round,pad=0.2')
        )

        # --- Append marker symbol right after label text ---
        ax1.scatter(
            fixed_label_x + 0.12, final_label_y,  # shift marker further right of text
            alpha=0.9, s=40, color=model_color_map[model],
            # marker='o',
            marker=model_markers[model],
            zorder=6
            # edgecolors='black', linewidth=0.8
        )

    # --- Axis formatting ---
    # breakpoint()
    dataset_labels = [f"{re.sub(r'(?<!^)(?=[A-Z])', '\n', dataset.replace('.json', '').replace('_', ''))}\n(n={dataset_size})" for dataset,dataset_size in zip(datasets,dataset_sizes)]
    
    dataset_labels.append("Means")
    # breakpoint()
    ax1.set_xticks(positions + [mean_x_pos])
    ax1.set_xticklabels(dataset_labels, fontsize=9)
    ax1.set_ylabel('ARI Score')
    ax1.grid(False)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # plt.figure(figsize=(12, 8))  # width=6 inches, height=4 inches
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Figure_Unsupervised.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(save_dir, 'Figure_Unsupervised.pdf'), dpi=300, bbox_inches='tight', facecolor='white')



def main():
    """Main function to generate all ARI-focused plots"""
    # Load data
    summary_data = load_summary_data('Artifacts/Group/Summary.json')
    model_df, dataset_df, replicate_df = extract_ari_data(summary_data)
    dataset_path = 'Datasets/Groupings'

    # Create figures directory if it doesn't exist
    save_dir = 'Figures'
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot clustering balance
    plot_group_sizes(dataset_path,save_dir)

    print("📊 Generating ARI-focused visualizations...")
    print(f"📁 Saving figures to: {save_dir}/")
    
    print("2️⃣ Creating Dataset Size Effect Plot...")
    plot_dataset_size_effect(dataset_df, replicate_df, save_dir)
    # --- AMI: identical plot alongside ARI ---
    ami_model_df, ami_dataset_df, ami_replicate_df = extract_ami_data(summary_data)
    print("2️⃣b Creating Dataset Size Effect Plot (AMI)...")
    plot_dataset_size_effect_ami(ami_dataset_df, ami_replicate_df, save_dir)
    
    # print("4️⃣ Creating ARI Distribution Plots...")
    # plot_ari_distributions(replicate_df, save_dir)
    
    print("\n✅ All plots generated successfully!")
    print(f"📈 Generated {len(os.listdir(save_dir))} files in {save_dir}/")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Models analyzed: {len(model_df)}")
    print(f"Datasets analyzed: {len(dataset_df)}")
    print(f"Total replicates: {len(replicate_df)}")
    # breakpoint()
    print(f"ARI range: {replicate_df['ari_score'].min():.3f} - {replicate_df['ari_score'].max():.3f}")
    print(f"Mean ARI: {replicate_df['ari_score'].mean():.3f} ± {replicate_df['ari_score'].std():.3f}")


    # Identify missing combinations of model, dataset, and replicate
    import pandas as pd
    from itertools import product

    # Extract unique identifiers
    models = model_df['model'].unique()
    datasets = dataset_df['dataset'].unique()
    replicates = replicate_df['replicate_id'].unique()

    # Create all possible combinations
    all_combinations = pd.DataFrame(list(product(models, datasets, replicates)), columns=['model', 'dataset', 'replicate_id'])

    # Merge with replicate_df to find missing combinations
    merged = pd.merge(all_combinations, replicate_df[['model', 'dataset', 'replicate_id']], on=['model', 'dataset', 'replicate_id'], how='left', indicator=True)
    missing_combinations = merged[merged['_merge'] == 'left_only'][['model', 'dataset', 'replicate_id']]

    # Print missing combinations
    print("\n❌ MISSING COMBINATIONS BASED ON COMPLETE CASES ANALYSIS:")
    if missing_combinations.empty:
        print("No missing combinations. All model-dataset-replicate cases are present.")
    else:
        print(missing_combinations.to_string(index=False))

if __name__ == "__main__":
    main()