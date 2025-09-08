import json
import os
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import numpy as np
from scripts.plot_groups import main as plot_groups
from scripts.helper import get_model_shorthand, get_model_colors
from time import sleep
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from sklearn.metrics import adjusted_mutual_info_score


def robust_adjusted_rand_score(
    taxon_groups_dict: Dict[str, str],
    corrected_taxon_to_generated: Dict[str, str]
) -> Tuple[Any, float, List[int], List[int], List[str]]:
    """
    Compute robust adjusted Rand index (AR†) in two ways:

    1) On the intersection of taxa (common_taxon) -> AR_dagger_common.
    2) Coverage-weighted AR† on the reference set:
       AR_dagger_all = AR_dagger_common * (coverage ** alpha)
       where coverage = |intersection| / |reference| and alpha = 1.0.

    This ensures missing taxa can never inflate the score.

    Returns
    -------
    AR_dagger_common : float or None
        Robust AR† on the common taxa; None if there are no common taxa.
    AR_dagger_all : float
        Coverage-weighted AR† (alpha=1.0), 0.0 if no common taxa or no reference taxa.
    reference_labels_common : list[int]
        Encoded reference labels for common taxa.
    generated_labels_common : list[int]
        Encoded generated labels for common taxa.
    common_taxon : list[str]
        Sorted list of taxa used for the "common" computation.
    """

    def _robust_ar_from_labels(ref_labels: List[int], gen_labels: List[int]) -> float:
        C = contingency_matrix(ref_labels, gen_labels)
        n = np.sum(C)
        if n <= 1:
            return 0.0
        row_sums = np.sum(C, axis=1)
        col_sums = np.sum(C, axis=0)
        P = np.sum(row_sums * (row_sums - 1)) / 2
        Q = np.sum(col_sums * (col_sums - 1)) / 2
        N = n * (n - 1) / 2

        Aw_list = []
        for i in range(C.shape[0]):
            ni_plus = row_sums[i]
            Pi = ni_plus * (ni_plus - 1) / 2
            if Pi == 0:
                continue
            T_i = np.sum(C[i, :] * (C[i, :] - 1) / 2)
            denom = Pi * (N - Q)
            Aw_list.append((N * T_i - Pi * Q) / denom if denom != 0 else 0.0)

        Av_list = []
        for j in range(C.shape[1]):
            n_plus_j = col_sums[j]
            Qj = n_plus_j * (n_plus_j - 1) / 2
            if Qj == 0:
                continue
            T_j = np.sum(C[:, j] * (C[:, j] - 1) / 2)
            denom = Qj * (N - P)
            Av_list.append((N * T_j - Qj * P) / denom if denom != 0 else 0.0)

        AW_dagger = float(np.mean(Aw_list)) if Aw_list else 0.0
        AV_dagger = float(np.mean(Av_list)) if Av_list else 0.0
        return (2 * AW_dagger * AV_dagger) / (AW_dagger + AV_dagger) if (AW_dagger + AV_dagger) != 0 else 0.0

    # 1) AR† on common taxa (intersection)
    common_taxon_set: Set[str] = set(taxon_groups_dict) & set(corrected_taxon_to_generated)
    common_taxon: List[str] = sorted(common_taxon_set)

    reference_labels_common: List[int] = []
    generated_labels_common: List[int] = []
    AR_dagger_common: Any = None

    if common_taxon:
        # Build label encodings for reference and generated groups present (global maps are fine)
        ref_groups = sorted(set(taxon_groups_dict.values()))
        gen_groups = sorted(set(corrected_taxon_to_generated.values()))
        ref_group_to_id = {group: i for i, group in enumerate(ref_groups)}
        gen_group_to_id = {group: i for i, group in enumerate(gen_groups)}

        for taxon in common_taxon:
            ref_group = taxon_groups_dict[taxon]
            gen_group = corrected_taxon_to_generated[taxon]
            reference_labels_common.append(ref_group_to_id[ref_group])
            generated_labels_common.append(gen_group_to_id[gen_group])

        AR_dagger_common = _robust_ar_from_labels(reference_labels_common, generated_labels_common)

    # 2) Coverage-weighted AR† (no union imputation, strictly penalizes missing taxa)
    #    coverage = |intersection| / |reference|
    alpha = 1.0  # increase to 2.0 for a harsher penalty if desired
    ref_count = len(taxon_groups_dict)
    coverage = (len(common_taxon) / ref_count) if ref_count > 0 else 0.0
    overlap_score = float(AR_dagger_common) if AR_dagger_common is not None else 0.0
    AR_dagger_all = overlap_score * (coverage ** alpha)

    return AR_dagger_common, AR_dagger_all, reference_labels_common, generated_labels_common, common_taxon

def plot_ami_vs_ari(summary_file: str = 'Artifacts/Group/Summary.json',
                    min_points_for_legend: int = 1,
                    alpha: float = 0.7,
                    s: int = 28):
    """
    Create a scatter plot of AMI (x) vs. ARI (y) for all replicates and draw a dashed 1:1 line.
    Saves PNG and PDF to Figures/Figure_AMI_vs_ARI.*

    Parameters
    ----------
    summary_file : str
        Path to the Summary.json produced by this script.
    min_points_for_legend : int
        Minimum number of points a model must have to appear in the legend.
    alpha : float
        Point transparency.
    s : int
        Point size.
    """
    import json
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Summary file not found: {summary_file}")
        return

    # Defensive checks for expected structure
    detailed = data.get("detailed_results", {})
    if not isinstance(detailed, dict) or not detailed:
        print("⚠️ No detailed_results found or empty in summary file—nothing to plot.")
        return

    model_to_points = {}
    for dataset, dct in detailed.items():
        models = dct.get("models", {})
        for model, mdata in models.items():
            for rep in mdata.get("replicates", []):
                ami = (rep.get("adjusted_mutual_info") or {}).get("score", None)
                ari = (rep.get("adjusted_rand_score") or {}).get("score", None)
                # Only plot when both are present and numeric
                if ami is None or ari is None:
                    continue
                try:
                    ami_val = float(ami)
                    ari_val = float(ari)
                except Exception:
                    continue
                model_to_points.setdefault(model, []).append((ami_val, ari_val))

    if not model_to_points:
        print("⚠️ No AMI/ARI pairs found—plot not created.")
        return

    # Colors and short labels
    try:
        color_map = get_model_colors(list(model_to_points.keys()))  # expects dict: {model_name: color}
    except Exception:
        color_map = {}

    # Build the plot
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_facecolor('white')

    # Plot per-model to get legend entries; use shorthand labels if available
    handles = []
    labels = []
    for model, pts in sorted(model_to_points.items(), key=lambda kv: kv[0].lower()):
        if len(pts) < 1:
            continue
        pts = np.array(pts, dtype=float)
        color = color_map.get(model, None)
        try:
            short = get_model_shorthand(model)
        except Exception:
            short = model
        sc = ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=alpha, label=short, color=color, edgecolors='none')
        if len(pts) >= min_points_for_legend:
            handles.append(sc)
            labels.append(short)
    # Axes settings
    ax.set_xlabel('Adjusted Mutual Information (AMI)')
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal', adjustable='box')

    # 1:1 reference line
    ax.plot([0, 1], [0, 1], linestyle='--', color='k', linewidth=1)

    # Grid & legend
    ax.set_axisbelow(True)
    ax.grid(True, which='both', axis='both', color='gray', alpha=0.3)
    if handles:
        # Place legend outside right
        ax.legend(handles, labels, title='Model', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # Save
    os.makedirs('Figures', exist_ok=True)
    out_png = os.path.join('Figures', 'Figure_AMI_vs_ARI.png')
    out_pdf = os.path.join('Figures', 'Figure_AMI_vs_ARI.pdf')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ AMI vs. ARI plot saved to:\n  - {out_png}\n  - {out_pdf}")


def compute_adjusted_mutual_info(
    taxon_groups_dict,
    generated_groups_data,
    *,
    alpha: float = 1.0,
    average_method: str = "arithmetic",
):
    """
    Compute Adjusted Mutual Information (AMI) in a way consistent with the robust ARI:
    - AMI_common: AMI on the intersection of taxa (no imputation).
    - AMI_all: coverage-weighted AMI_common, where coverage = |intersection| / |reference|.
    """
    from difflib import get_close_matches

    # 1) Build taxon->generated-group mapping from generated groups
    taxon_to_generated = {}
    for group in generated_groups_data:
        gname = group['group_name']
        for taxon in group.get('taxon', []):
            taxon_to_generated[taxon] = gname

    # 2) Fuzzy-correct taxa missing in the reference dict (same logic as ARI)
    corrected = {}
    fuzzy_fix_count = 0
    for taxon, g in taxon_to_generated.items():
        if taxon not in taxon_groups_dict:
            match = get_close_matches(taxon, taxon_groups_dict.keys(), n=1, cutoff=0.85)
            if match:
                corrected[match[0]] = g
                fuzzy_fix_count += 1
            else:
                corrected[taxon] = g
        else:
            corrected[taxon] = g

    # 3) Reuse the robust ARI function only to obtain the aligned label vectors + common taxon
    #    (you just extended it to return labels too)
    try:
        _, _, ref_labels_common, gen_labels_common, common_taxon = robust_adjusted_rand_score(
            taxon_groups_dict, corrected
        )
    except Exception:
        return None, "No common taxon found", {}

    # 4) Coverage and AMI
    ref_count = len(taxon_groups_dict) or 0
    coverage = (len(common_taxon) / ref_count) if ref_count > 0 else 0.0

    if len(common_taxon) == 0:
        ami_common = None
        ami_all = 0.0
        info = "No common taxon found"
    else:
        ami_common = adjusted_mutual_info_score(
            ref_labels_common, gen_labels_common, average_method=average_method
        )
        ami_all = float(ami_common) * (coverage ** alpha)
        info = f"Computed on {len(common_taxon)} common taxon"

    # 5) Stats (mirror ARI stats + add coverage)
    ref_groups = list(set(taxon_groups_dict.values()))
    gen_groups = list(set(corrected.values()))
    stats = {
        "total_reference_taxon": len(taxon_groups_dict),
        "total_generated_taxon": len(taxon_to_generated),
        "common_taxon_count": len(common_taxon),
        "reference_groups_count": len(ref_groups),
        "generated_groups_count": len(gen_groups),
        "fuzzy_fix_count": fuzzy_fix_count,
        "coverage": coverage,
    }

    return ami_all, info, stats




from collections import defaultdict
from typing import Dict

def pairwise_cluster_f1(
    gold_map: Dict[str, str],
    pred_map: Dict[str, str],
    *,
    evaluation_universe: str = "reference",  # "reference" (default) or "union"
    ignore_singleton_gold_for_macro: bool = True
):
    """
    Pairwise clustering evaluation (name-agnostic):
      - Positives  = unordered pairs co-clustered in gold.
      - Predicted  = unordered pairs co-clustered in prediction.
      - TP         = pairs co-clustered in both.
      - FN         = gold pairs that are not predicted co-clustered.
      - FP         = predicted pairs that are not gold co-clustered.

    Missing taxa handling:
      - If a gold taxon is missing from prediction, it is treated as a
        singleton predicted cluster => it adds FNs (cannot form predicted pairs).
      - Predicted-only taxa:
          * evaluation_universe="reference" (default): ignored (no FP penalty).
          * evaluation_universe="union": included; assigned unique gold singletons
            so their within-predicted pairs contribute to FP.

    Macro averaging:
      - Per-gold-cluster pairwise precision/recall/F1 is computed over pairs
        inside each gold cluster. By default, singleton gold clusters are
        excluded from macro because they have zero positive pairs.

    Returns
    -------
    dict with keys:
      - "micro": global pair counts & P/R/F1 (dominated by large clusters)
      - "macro": average of per-gold-cluster metrics (treats clusters equally)
      - "per_gold_cluster": detailed per-cluster counts and metrics
    """
    def _choose2(x: int) -> int:
        return x * (x - 1) // 2 if x >= 2 else 0

    # 1) Establish taxa for evaluation
    gold_taxa = set(gold_map)
    pred_taxa = set(pred_map)
    if evaluation_universe == "reference":
        taxa = sorted(gold_taxa)
    elif evaluation_universe == "union":
        taxa = sorted(gold_taxa | pred_taxa)
    else:
        raise ValueError("evaluation_universe must be 'reference' or 'union'")

    # 2) Build label assignments over evaluation taxa
    GOLD_SINGLETON_PREFIX = "__G_SINGLETON__:"
    PRED_SINGLETON_PREFIX = "__P_SINGLETON__:"

    gold_label = {}
    pred_label = {}
    for t in taxa:
        if t in gold_map:
            gold_label[t] = ("G", gold_map[t])  # namespaced gold label
        else:
            # union-mode only: predicted-only taxa become gold singletons
            gold_label[t] = ("G", f"{GOLD_SINGLETON_PREFIX}{t}")

        if t in pred_map:
            pred_label[t] = ("P", pred_map[t])  # namespaced predicted label
        else:
            # reference-mode: missing prediction => predicted singleton
            pred_label[t] = ("P", f"{PRED_SINGLETON_PREFIX}{t}")

    # 3) Group taxa by gold/predicted clusters over evaluation universe
    gold_clusters = defaultdict(list)
    pred_clusters = defaultdict(list)
    for t in taxa:
        gold_clusters[gold_label[t]].append(t)
        pred_clusters[pred_label[t]].append(t)

    n_gold = {k: len(v) for k, v in gold_clusters.items()}
    n_pred = {j: len(v) for j, v in pred_clusters.items()}

    # 4) Cross-tab counts c_{k,j} (how many from gold cluster k fell into predicted cluster j)
    c = defaultdict(int)
    for j, items in pred_clusters.items():
        tmp = defaultdict(int)
        for t in items:
            tmp[gold_label[t]] += 1
        for k, cnt in tmp.items():
            c[(k, j)] = cnt

    # 5) Global (micro) pair counts
    TP = sum(_choose2(cnt) for cnt in c.values())
    gold_pairs = sum(_choose2(sz) for sz in n_gold.values())
    pred_pairs = sum(_choose2(sz) for sz in n_pred.values())
    FN = gold_pairs - TP
    FP = pred_pairs - TP

    prec_micro = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_micro  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_micro   = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0

    # 6) Per-gold-cluster metrics for macro averaging
    per_cluster = {}
    for k, nk in n_gold.items():
        # Optionally skip singletons (no positive pairs)
        if ignore_singleton_gold_for_macro and nk < 2:
            continue

        # TP_k: correctly co-clustered pairs inside gold cluster k
        TP_k = sum(_choose2(c.get((k, j), 0)) for j in n_pred.keys())
        FN_k = _choose2(nk) - TP_k

        # FP_k: predicted pairs that incorrectly tie members of k with outsiders
        FP_k = sum(
            c.get((k, j), 0) * (n_pred[j] - c.get((k, j), 0))
            for j in n_pred.keys()
        )

        p_k = TP_k / (TP_k + FP_k) if (TP_k + FP_k) > 0 else 0.0
        r_k = TP_k / (TP_k + FN_k) if (TP_k + FN_k) > 0 else 0.0
        f1_k = (2 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0

        # drop namespace in key for readability
        per_cluster[k[1]] = {
            "size": nk, "TP": TP_k, "FP": FP_k, "FN": FN_k,
            "precision": p_k, "recall": r_k, "f1": f1_k,
        }

    macro_precision = (
        sum(v["precision"] for v in per_cluster.values()) / len(per_cluster)
        if per_cluster else 0.0
    )
    macro_recall = (
        sum(v["recall"] for v in per_cluster.values()) / len(per_cluster)
        if per_cluster else 0.0
    )
    macro_f1 = (
        sum(v["f1"] for v in per_cluster.values()) / len(per_cluster)
        if per_cluster else 0.0
    )

    return {
        "micro": {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": prec_micro, "recall": rec_micro, "f1": f1_micro,
            "gold_pairs": gold_pairs, "pred_pairs": pred_pairs,
            "n_eval_taxa": len(taxa),
            "universe": evaluation_universe,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "n_clusters": len(per_cluster),
            "ignore_singleton_gold_for_macro": ignore_singleton_gold_for_macro,
        },
        "per_gold_cluster": per_cluster,
        "counts": {
            "n_ref_taxa": len(gold_taxa),
            "n_pred_taxa": len(pred_taxa),
            "n_union_taxa": len(gold_taxa | pred_taxa),
            "n_missing_taxa": len(gold_taxa - pred_taxa),  # counted as FN
            "n_extra_taxa": len(pred_taxa - gold_taxa),    # only penalized if universe='union'
        }
    }



def load_taxon_groups(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return data  # Return the full taxon-to-group mapping

def get_unique_groups(taxon_groups_dict):
    """Extract unique group names from taxon-to-group mapping"""
    return list(set(taxon_groups_dict.values()))

def load_functional_groups(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        return json.load(file)

def compute_similarity(input_groups, generated_groups, threshold=0.7):
    strong_matches = []
    unmatched_input = []
    unmatched_generated = set(generated_groups)

    for input_group in input_groups:
        scores = []
        for gen_group in generated_groups:
            vectorizer = TfidfVectorizer().fit([input_group, gen_group])
            tfidf = vectorizer.transform([input_group, gen_group])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            scores.append((gen_group, score))
        try:
            best_match = max(scores, key=lambda x: x[1])
        except:
            breakpoint()
        if best_match[1] >= threshold:
            strong_matches.append((input_group, best_match[0], round(best_match[1], 2)))
            unmatched_generated.discard(best_match[0])
        else:
            unmatched_input.append(input_group)

    return strong_matches, unmatched_input, list(unmatched_generated)

def create_taxon_to_generated_group_mapping(generated_groups_data):
    """
    Create a mapping from taxon to generated functional groups.
    Now we can use the taxon lists directly from the generated groups.
    """
    taxon_to_generated = {}
    
    for group in generated_groups_data:
        group_name = group['group_name']
        taxon_list = group.get('taxon', [])
        
        for taxon in taxon_list:
            taxon_to_generated[taxon] = group_name
    
    return taxon_to_generated

def compute_adjusted_rand_score(taxon_groups_dict, generated_groups_data):
    """
    Compute adjusted rand score between reference and generated groupings,
    applying fuzzy matching to taxon not found in the reference dictionary.
    """

    from difflib import get_close_matches

    # Create taxon-to-generated-group mapping
    taxon_to_generated = create_taxon_to_generated_group_mapping(generated_groups_data)

    # Apply fuzzy matching for taxon not in taxon_groups_dict
    corrected_taxon_to_generated = {}
    fuzzy_fix_count = 0

    for taxon, group in taxon_to_generated.items():
        if taxon not in taxon_groups_dict:
            match = get_close_matches(taxon, taxon_groups_dict.keys(), n=1, cutoff=0.85)
            if match:
                corrected_taxon_to_generated[match[0]] = group
                fuzzy_fix_count += 1
            else:
                corrected_taxon_to_generated[taxon] = group
        else:
            corrected_taxon_to_generated[taxon] = group

    # Compute adjusted rand score
    try:
        ari_score_robust,ari_score_robust_all, reference_labels_common, generated_labels_common, common_taxon = robust_adjusted_rand_score(taxon_groups_dict, corrected_taxon_to_generated)
        
        pairwise = pairwise_cluster_f1(
            taxon_groups_dict, corrected_taxon_to_generated,
            evaluation_universe="reference",           # counts missing taxa as FN; ignores predicted-only taxa
            ignore_singleton_gold_for_macro=True       # exclude singletons from macro (they have 0 positive pairs)
        )

        print("Pairwise micro-F1:", pairwise["micro"]["f1"])
        print("Pairwise macro-F1:", pairwise["macro"]["f1"])

        # ari_score_robust,ref_groups,gen_groups, common_taxon,f1_score = robust_adjusted_rand_score_with_f1(taxon_groups_dict, corrected_taxon_to_generated)
    except:
        return None, "No common taxon found", {}
    # ari_score = pairwise["macro"]["f1"]
    ari_score = ari_score_robust_all
        # Create label encoders
    ref_groups = list(set(taxon_groups_dict.values()))
    gen_groups = list(set(corrected_taxon_to_generated.values()))
    # breakpoint()
    # Additional statistics
    stats = {
        "total_reference_taxon": len(taxon_groups_dict),
        "total_generated_taxon": len(taxon_to_generated),
        "common_taxon_count": len(common_taxon),
        "reference_groups_count": len(ref_groups),
        "generated_groups_count": len(gen_groups),
        "fuzzy_fix_count": fuzzy_fix_count
    }

    return ari_score, f"Computed on {len(common_taxon)} common taxon", stats

def analyze_group_sizes(summary_file='Artifacts/Group/Summary.json'):
    sleep(2)
    # Load the JSON data from the file
    with open(summary_file, "r") as f:
        data = json.load(f)
    # Dictionary to collect statistics for each dataset and model
    detailed_statistics = {}

    # Dictionary to collect overall statistics for datasets and models
    dataset_stats = {}
    model_stats = {}

    # Traverse the nested structure
    for dataset, dataset_content in data.get("detailed_results", {}).items():
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {
                "taxon_diff": [],
                "groups_diff": [],
                "fuzzy_fix_count": [],
                "fuzzy_ratio": []  
            }

        for model, model_content in dataset_content.get("models", {}).items():
            model_key = model
            if model_key not in model_stats:
                model_stats[model_key] = {
                    "taxon_diff": [],
                    "groups_diff": [],
                    "fuzzy_fix_count": [],
                    "fuzzy_ratio": []  
                }

            key = f"{dataset}__{model}"
            detailed_statistics[key] = {}

            for replicate in model_content['replicates']:
                try:
                    stats = replicate.get("adjusted_rand_score", {}).get("statistics", {})
                except:
                    continue
                for stat_key, value in stats.items():
                    if stat_key not in detailed_statistics[key]:
                        detailed_statistics[key][stat_key] = []
                    detailed_statistics[key][stat_key].append(value)

                ref_taxon = stats.get("total_reference_taxon", 0)
                gen_taxon = stats.get("common_taxon_count", 0)
                ref_groups = stats.get("reference_groups_count", 0)
                gen_groups = stats.get("generated_groups_count", 0)
                fuzzy_fix = stats.get("fuzzy_fix_count", 0)

                try:
                    taxon_diff = (gen_taxon - ref_taxon)/ref_taxon
                    groups_diff = (gen_groups - ref_groups)/ref_groups
                    fuzzy_ratio = fuzzy_fix/ref_taxon
                except ZeroDivisionError:
                    taxon_diff = 0
                    groups_diff = 0
                    fuzzy_ratio = 0

                dataset_stats[dataset]["taxon_diff"].append(taxon_diff)
                dataset_stats[dataset]["groups_diff"].append(groups_diff)
                dataset_stats[dataset]["fuzzy_fix_count"].append(fuzzy_fix)
                dataset_stats[dataset]["fuzzy_ratio"].append(fuzzy_ratio)

                model_stats[model_key]["taxon_diff"].append(taxon_diff)
                model_stats[model_key]["groups_diff"].append(groups_diff)
                model_stats[model_key]["fuzzy_fix_count"].append(fuzzy_fix)
                model_stats[model_key]["fuzzy_ratio"].append(fuzzy_ratio)


    # Compute mean and standard deviation for each dataset-model pair
    detailed_summary = {
        key: {
            stat: {
                "mean": np.mean(values),
                "std_dev": np.std(values)
            }
            for stat, values in stat_dict.items()
        }
        for key, stat_dict in detailed_statistics.items()
    }

        
    def plot_model_summary(model_summary, model_stats):
        sorted_models = sorted(
            model_summary.keys(),
            key=lambda m: (
                len(model_stats[m]['taxon_diff']),
                model_summary[m]['taxon_diff']['mean'],
                model_summary[m]['groups_diff']['mean'],
                np.mean(model_stats[m]['fuzzy_ratio'])
            )
        )

        if 'ollama_hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL' in sorted_models:
            sorted_models.remove('ollama_hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL')

        member_counts = [len(model_stats[model]['taxon_diff']) for model in sorted_models]
        short_labels = [get_model_shorthand(model) for model in sorted_models]
        taxon_means = [model_summary[model]['taxon_diff']['mean'] for model in sorted_models]
        taxon_stds = [model_summary[model]['taxon_diff']['std_dev'] for model in sorted_models]
        groups_means = [model_summary[model]['groups_diff']['mean'] for model in sorted_models]
        groups_stds = [model_summary[model]['groups_diff']['std_dev'] for model in sorted_models]
        fuzzy_stds = [model_summary[model]['fuzzy_ratio']['mean'] for model in sorted_models]
        fuzzy_ratio = [model_summary[model]['fuzzy_ratio']['std_dev'] for model in sorted_models]

        x = np.arange(len(sorted_models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')

        bars1 = ax.bar(x - width, taxon_means, width, label='Missing Taxa', capsize=5)
        ax.errorbar(x - width, taxon_means, yerr=taxon_stds, fmt='none', ecolor='black', alpha=0.5, capsize=2, elinewidth=1  )

        bars2 = ax.bar(x, groups_means, width, label='Group Number', capsize=5)
        ax.errorbar(x, groups_means, yerr=groups_stds, fmt='none', ecolor='black', alpha=0.5, capsize=2, elinewidth=1  )

        bars3 = ax.bar(x + width, -1 * np.array(fuzzy_ratio), width, label='Misspelled Taxa', capsize=2)
        ax.errorbar(x + width, -1 * np.array(fuzzy_ratio), yerr=fuzzy_stds, fmt='none', ecolor='black', alpha=0.5, capsize=5, elinewidth=1  )



        for i, count in enumerate(member_counts):
            ax.text(x[i]- 0.2, 0.05, f'{count}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Proportion Relative to Gold Standard')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45)
        ax.legend()

        # ✅ Ensure horizontal gridlines are visible and styled
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='gray', alpha=0.3)

        plt.tight_layout()

        os.makedirs('Figures', exist_ok=True)
        plt.savefig(os.path.join('Figures', 'Figure_Group_Missing.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join('Figures', 'Figure_Group_Missing.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        # plt.show()




    dataset_summary = {
        dataset: {
            "taxon_diff": {
                "mean": np.mean(stats["taxon_diff"]),
                "std_dev": np.std(stats["taxon_diff"])
            },
            "groups_diff": {
                "mean": np.mean(stats["groups_diff"]),
                "std_dev": np.std(stats["groups_diff"])
            },
            "fuzzy_fix_count": {
                "mean": np.mean(stats["fuzzy_fix_count"]),
                "std_dev": np.std(stats["fuzzy_fix_count"])
            },
            "fuzzy_ratio": {
                "mean": np.mean(stats["fuzzy_ratio"]),
                "std_dev": np.std(stats["fuzzy_ratio"])
            }
        }
        for dataset, stats in dataset_stats.items()
    }

    model_summary = {
        model: {
            "taxon_diff": {
                "mean": np.mean(stats["taxon_diff"]),
                "std_dev": np.std(stats["taxon_diff"])
            },
            "groups_diff": {
                "mean": np.mean(stats["groups_diff"]),
                "std_dev": np.std(stats["groups_diff"])
            },
            "fuzzy_fix_count": {
                "mean": np.mean(stats["fuzzy_fix_count"]),
                "std_dev": np.std(stats["fuzzy_fix_count"])
            },
            "fuzzy_ratio": {
                "mean": np.mean(stats["fuzzy_ratio"]),
                "std_dev": np.std(stats["fuzzy_ratio"])
            }
        }
        for model, stats in model_stats.items()
    }

    plot_model_summary(model_summary, model_stats)
    output = {
        "dataset_model_statistics": detailed_summary,
        "dataset_statistics": dataset_summary,
        "model_statistics": model_summary
    }

    output_path = os.path.join(os.path.dirname(summary_file), "Groups.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Analysis complete. Results saved to {output_path}")



    # breakpoint()


def main():
    dataset_path = 'Datasets/Groupings'
    groupings_file = 'Artifacts/Group/Output-Raw.json'
    output_file = 'Artifacts/Group/Summary.json'

    all_groupings = load_functional_groups(groupings_file)
    taxon_files = [os.path.join(root, f) for root, _, files in os.walk(dataset_path) for f in files]

    # Dictionary to store all results
    summary_results = {}

    for taxon_file in taxon_files:
        taxon_key = os.path.basename(taxon_file)
        taxon_groups_dict = load_taxon_groups(taxon_file)
        unique_input_groups = get_unique_groups(taxon_groups_dict)

        print(f"\n=== Dataset: {taxon_key} ===")
        print(f"Reference dataset contains {len(taxon_groups_dict)} taxon in {len(unique_input_groups)} groups")

        # Initialize dataset results
        summary_results[taxon_key] = {
            "reference_info": {
                "total_taxon": len(taxon_groups_dict),
                "total_groups": len(unique_input_groups),
                "groups": unique_input_groups,
                "dataset_size": len(taxon_groups_dict)
            },
            "models": {}
        }

        for model, replicates_list in all_groupings.get(taxon_key, {}).items():
            print(f"\n--- Model: {model} ---")
            print(f"Found {len(replicates_list)} replicates")

            # Initialize model results
            summary_results[taxon_key]["models"][model] = {
                "replicates": []
            }

            for replicate_idx, replicate_data in enumerate(replicates_list):
                print(f"\n  Replicate {replicate_idx + 1}:")

                # Check for placeholder or error
                if replicate_data.get("placeholder", False) or "error" in replicate_data:
                    print("  ⚠️ Placeholder or error detected. Skipping analysis.")
                    replicate_result = {
                        "replicate_id": replicate_idx + 1,
                        "generated_groups": None,
                        "similarity_analysis": None,
                        "adjusted_rand_score": None,
                        "error": replicate_data.get("error", "Placeholder entry")
                    }
                    summary_results[taxon_key]["models"][model]["replicates"].append(replicate_result)
                    continue

                generated_groups_data = replicate_data.get("functional_groups", [])
                generated_group_names = [group['group_name'] for group in generated_groups_data]

                print(f"  Generated {len(generated_group_names)} functional groups")

                # strong_matches, unmatched_input, unmatched_generated = compute_similarity(
                #     unique_input_groups, generated_group_names
                # )

                # print("\n  📊 GROUP NAME SIMILARITY ANALYSIS:")
                # print("  Strong Matches (similarity >= 0.7):")
                # for input_group, matched_group, score in strong_matches:
                #     print(f"    - '{input_group}' ↔ '{matched_group}' (score: {score})")

                # print("\n  Unmatched Input Groups:")
                # for group in unmatched_input:
                #     print(f"    - {group}")

                # print("\n  Unmatched Generated Groups:")
                # for group in unmatched_generated:
                #     print(f"    - {group}")

                print("\n  🎯 ADJUSTED RAND SCORE ANALYSIS:")
                ari_score, ari_info, ari_stats = compute_adjusted_rand_score(taxon_groups_dict, generated_groups_data)
                # AMI (Adjusted Mutual Information) analysis
                ami_score, ami_info, ami_stats = compute_adjusted_mutual_info(
                    taxon_groups_dict, generated_groups_data
                )
                interpretation = None
                if ari_score is not None:
                    # print(f"  Adjusted Rand Score: {ari_score:.4f}")
                    # print(f"  Info: {ari_info}")

                    if ari_score >= 0.8:
                        interpretation = "Excellent agreement"
                    elif ari_score >= 0.6:
                        interpretation = "Good agreement"
                    elif ari_score >= 0.4:
                        interpretation = "Moderate agreement"
                    elif ari_score >= 0.2:
                        interpretation = "Fair agreement"
                    else:
                        interpretation = "Poor agreement"

                    # print(f"  Interpretation: {interpretation}")
                    # print(f"  Statistics: {ari_stats}")
                else:
                    print(f"  Could not compute ARI: {ari_info}")

                replicate_result = {
                    "replicate_id": replicate_idx + 1,
                    "generated_groups": {
                        "total_groups": len(generated_group_names),
                        "group_names": generated_group_names,
                        "groups_with_taxon": [
                            {
                                "group_name": group['group_name'],
                                "taxon_count": len(group.get('taxon', [])),
                                "taxon": group.get('taxon', [])
                            }
                            for group in generated_groups_data
                        ]
                    },
                    # "similarity_analysis": {
                    #     "strong_matches": [
                    #         {
                    #             "input_group": input_group,
                    #             "matched_group": matched_group,
                    #             "similarity_score": score
                    #         }
                    #         for input_group, matched_group, score in strong_matches
                    #     ],
                    #     "unmatched_input_groups": unmatched_input,
                    #     "unmatched_generated_groups": unmatched_generated,
                    #     "match_rate": len(strong_matches) / len(unique_input_groups) if unique_input_groups else 0
                    # },
                    "adjusted_rand_score": {
                        "score": ari_score,
                        "info": ari_info,
                        "interpretation": interpretation,
                        "statistics": ari_stats
                    },

                    # NEW: store AMI next to ARI (same shape)
                    "adjusted_mutual_info": {
                        "score": ami_score,
                        "info": ami_info,
                        "statistics": ami_stats
                    }

                }

                summary_results[taxon_key]["models"][model]["replicates"].append(replicate_result)

            print("-" * 60)
    

    high_level_summary = {
        "overview": {
            "total_datasets": len(summary_results),
            "datasets_analyzed": list(summary_results.keys())
        },
        "model_performance": {},
        "dataset_summary": {}
    }

    # Aggregate statistics across all datasets and models
    all_models = set()
    for dataset_data in summary_results.values():
        all_models.update(dataset_data["models"].keys())

    # Calculate model performance summary
    for model in all_models:
        model_ari_scores = []
        model_ami_scores = []  # NEW
        model_match_rates = []
        model_replicate_count = 0
        datasets_with_model = 0

        for dataset, dataset_data in summary_results.items():
            if model in dataset_data["models"]:
                datasets_with_model += 1
                replicates = dataset_data["models"][model]["replicates"]
                model_replicate_count += len(replicates)

                for replicate in replicates:
                    if replicate.get("adjusted_rand_score") and replicate["adjusted_rand_score"]["score"] is not None:
                        model_ari_scores.append(replicate["adjusted_rand_score"]["score"])
                    if replicate.get("similarity_analysis") and replicate["similarity_analysis"].get("match_rate") is not None:
                        model_match_rates.append(replicate["similarity_analysis"]["match_rate"])

                if replicate.get("adjusted_mutual_info") and replicate["adjusted_mutual_info"]["score"] is not None:
                    model_ami_scores.append(replicate["adjusted_mutual_info"]["score"])




        model_ari_scores = np.array(model_ari_scores)
        model_ami_scores = np.array(model_ami_scores)  # NEW
        model_match_rates = np.array(model_match_rates)

        high_level_summary["model_performance"][model] = {
            "datasets_analyzed": int(datasets_with_model),
            "total_replicates": int(model_replicate_count),
            "adjusted_rand_score": {
                "count": int(np.count_nonzero(~np.isnan(model_ari_scores))),
                "mean": float(np.nanmean(model_ari_scores)) if model_ari_scores.size else None,
                "min": float(np.nanmin(model_ari_scores)) if model_ari_scores.size else None,
                "max": float(np.nanmax(model_ari_scores)) if model_ari_scores.size else None,
                "std": float(np.nanstd(model_ari_scores)) if model_ari_scores.size > 1 else None
            },

            "adjusted_mutual_info": {
                "count": int(np.count_nonzero(~np.isnan(model_ami_scores))),
                "mean": float(np.nanmean(model_ami_scores)) if model_ami_scores.size else None,
                "min": float(np.nanmin(model_ami_scores)) if model_ami_scores.size else None,
                "max": float(np.nanmax(model_ami_scores)) if model_ami_scores.size else None,
                "std": float(np.nanstd(model_ami_scores)) if model_ami_scores.size > 1 else None
            },

            "match_rate": {
                "mean": float(np.nanmean(model_match_rates)) if model_match_rates.size else None,
                "min": float(np.nanmin(model_match_rates)) if model_match_rates.size else None,
                "max": float(np.nanmax(model_match_rates)) if model_match_rates.size else None,
                "std": float(np.nanstd(model_match_rates)) if model_match_rates.size > 1 else None
            }
        }



    # Calculate dataset summary
    for dataset, dataset_data in summary_results.items():
        dataset_ari_scores = []
        dataset_ami_scores = []  # NEW
        dataset_match_rates = []
        total_replicates = 0
        models_analyzed = len(dataset_data["models"])

        for model_data in dataset_data["models"].values():
            replicates = model_data["replicates"]
            total_replicates += len(replicates)

            for replicate in replicates:
                if replicate.get("adjusted_rand_score") and replicate["adjusted_rand_score"]["score"] is not None:
                    dataset_ari_scores.append(replicate["adjusted_rand_score"]["score"])

                if replicate.get("adjusted_mutual_info") and replicate["adjusted_mutual_info"]["score"] is not None:
                    dataset_ami_scores.append(replicate["adjusted_mutual_info"]["score"])
                # match rate

                if replicate.get("similarity_analysis") and replicate["similarity_analysis"].get("match_rate") is not None:
                    dataset_match_rates.append(replicate["similarity_analysis"]["match_rate"])


        dataset_ari_scores = np.array(dataset_ari_scores)
        dataset_ami_scores = np.array(dataset_ami_scores)  # NEW
        dataset_match_rates = np.array(dataset_match_rates)

        high_level_summary["dataset_summary"][dataset] = {
            "dataset_size": int(dataset_data["reference_info"]["total_taxon"]),
            "reference_taxon_count": int(dataset_data["reference_info"]["total_taxon"]),
            "reference_groups_count": int(dataset_data["reference_info"]["total_groups"]),
            "models_analyzed": int(models_analyzed),
            "total_replicates": int(total_replicates),
            "adjusted_rand_score": {
                "count": int(np.count_nonzero(~np.isnan(dataset_ari_scores))),
                "mean": float(np.nanmean(dataset_ari_scores)) if dataset_ari_scores.size else None,
                "min": float(np.nanmin(dataset_ari_scores)) if dataset_ari_scores.size else None,
                "max": float(np.nanmax(dataset_ari_scores)) if dataset_ari_scores.size else None
            },

            "adjusted_mutual_info": {
                "count": int(np.count_nonzero(~np.isnan(dataset_ami_scores))),
                "mean": float(np.nanmean(dataset_ami_scores)) if dataset_ami_scores.size else None,
                "min": float(np.nanmin(dataset_ami_scores)) if dataset_ami_scores.size else None,
                "max": float(np.nanmax(dataset_ami_scores)) if dataset_ami_scores.size else None
            },

            "match_rate": {
                "mean": float(np.nanmean(dataset_match_rates)) if dataset_match_rates.size else None,
                "min": float(np.nanmin(dataset_match_rates)) if dataset_match_rates.size else None,
                "max": float(np.nanmax(dataset_match_rates)) if dataset_match_rates.size else None
            }
        }

    analyze_group_sizes()
    # Combine summary with detailed results
    final_output = {
        "summary": high_level_summary,
        "detailed_results": summary_results
    }

    # Write results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Analysis complete! Results saved to '{output_file}'")

    # Print high-level summary statistics
    print("\n📈 HIGH-LEVEL SUMMARY:")
    print(f"Total datasets analyzed: {high_level_summary['overview']['total_datasets']}")
    print(f"Datasets: {', '.join(high_level_summary['overview']['datasets_analyzed'])}")
    
    print("\n🤖 MODEL PERFORMANCE OVERVIEW:")
    for model, perf in high_level_summary["model_performance"].items():
        print(f"\n{model}:")
        print(f"  - Datasets analyzed: {perf['datasets_analyzed']}")
        print(f"  - Total replicates: {perf['total_replicates']}")
        
        ari_data = perf['adjusted_rand_score']
        if ari_data['mean'] is not None:
            std_val = ari_data['std'] if ari_data['std'] is not None else 0
            print(f"  - Adjusted Rand Score: {ari_data['mean']:.4f} ± {std_val:.4f}")
            print(f"    Range: {ari_data['min']:.4f} - {ari_data['max']:.4f} (n={ari_data['count']})")
        else:
            print(f"  - Adjusted Rand Score: N/A")
        
        match_data = perf['match_rate']
        if match_data['mean'] is not None:
            std_val = match_data['std'] if match_data['std'] is not None else 0
            print(f"  - Match Rate: {match_data['mean']:.2%} ± {std_val:.2%}")
            print(f"    Range: {match_data['min']:.2%} - {match_data['max']:.2%}")
    
    print("\n📊 DATASET SUMMARY:")
    for dataset, summary in high_level_summary["dataset_summary"].items():
        print(f"\n{dataset}:")
        print(f"  - Dataset Size: {summary['dataset_size']} taxon")
        print(f"  - Reference Groups: {summary['reference_groups_count']} functional groups")
        print(f"  - Models analyzed: {summary['models_analyzed']}")
        print(f"  - Total replicates: {summary['total_replicates']}")
        
        ari_data = summary['adjusted_rand_score']
        if ari_data['mean'] is not None:
            print(f"  - Average ARI: {ari_data['mean']:.4f} (range: {ari_data['min']:.4f}-{ari_data['max']:.4f}, n={ari_data['count']})")
        else:
            print(f"  - Average ARI: N/A")
        
        match_data = summary['match_rate']
        if match_data['mean'] is not None:
            print(f"  - Average Match Rate: {match_data['mean']:.2%} (range: {match_data['min']:.2%}-{match_data['max']:.2%})")

if __name__ == "__main__":
    main()
    plot_groups()
    plot_ami_vs_ari('Artifacts/Group/Summary.json')
