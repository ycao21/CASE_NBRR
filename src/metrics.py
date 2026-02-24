import torch
import numpy as np
from typing import Tuple, Dict, List, Optional


def compute_recall_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> Tuple[float, int]:
    """
    Compute Recall@K for a batch (used during training validation).
    
    Recall@K = (# relevant items in top-K) / (# total relevant items)
    
    This is a fast approximation suitable for training monitoring because:
    - Works on sampled negatives (not full vocabulary)
    - Uses logits/labels directly from loss function output
    - Returns single aggregated metric across batch
    
    Args:
        logits: Scores for items [B, N] where N = positives + sampled negatives
                Padding positions are marked with float("-inf")
        labels: Binary labels [B, N] where 1 = positive, 0 = negative/padding
        k: Number of top items to consider
    
    Returns:
        recall: Average recall@K across valid samples
        num_samples: Number of valid samples (with at least 1 positive)
        
    Example:
        >>> logits = torch.tensor([[0.9, 0.7, 0.3, 0.1, -inf],    # Sample 1: 2 positives
        ...                        [0.8, 0.6, 0.4, 0.2, 0.0]])    # Sample 2: 1 positive
        >>> labels = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0],
        ...                        [1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> recall, n = compute_recall_at_k(logits, labels, k=2)
        >>> # Sample 1: top-2 = [0.9, 0.7] → both are positives → recall = 2/2 = 1.0
        >>> # Sample 2: top-2 = [0.8, 0.6] → 1 is positive → recall = 1/1 = 1.0
        >>> # Average: (1.0 + 1.0) / 2 = 1.0
    """
    # Filter out samples with no valid candidates or no positives
    valid_candidates_mask = logits > float("-inf")
    num_valid_per_sample = valid_candidates_mask.sum(dim=1)
    sufficient_candidates = num_valid_per_sample >= k
    
    positives_per_sample = labels.sum(dim=1)
    valid_samples = (positives_per_sample > 0) & sufficient_candidates
    
    if valid_samples.sum() == 0:
        return 0.0, 0
    
    # Get top-K predictions (k might be larger than available items)
    k_actual = min(k, logits.size(1))
    # if k_actual < k:
    #     print(f"Warning: k_actual < k: {k_actual} < {k}")
    _, top_k_indices = torch.topk(logits, k=k_actual, dim=1)
    
    # Get labels for top-K items
    top_k_labels = torch.gather(labels, 1, top_k_indices)
    hits_per_sample = top_k_labels.sum(dim=1)
    
    # Compute recall per sample: hits / total_positives
    recall_denominator = torch.clamp(positives_per_sample, min=1.0)
    recall_per_sample = hits_per_sample / recall_denominator
    
    # Average over valid samples
    total_recall = (recall_per_sample * valid_samples.float()).sum().item()
    num_valid = valid_samples.sum().item()
    
    return total_recall / num_valid if num_valid > 0 else 0.0, num_valid


def compute_precision_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> Tuple[float, int]:
    """
    Compute Precision@K for a batch.
    
    Precision@K = (# relevant items in top-K) / K
    
    Args:
        logits: Scores for items [B, N] where N = positives + sampled negatives
        labels: Binary labels [B, N] where 1 = positive, 0 = negative
        k: Number of top items to consider
    
    Returns:
        precision: Average precision@K across valid samples
        num_samples: Number of valid samples
        
    Example:
        >>> logits = torch.tensor([[0.9, 0.7, 0.3, 0.1]])
        >>> labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        >>> precision, n = compute_precision_at_k(logits, labels, k=3)
        >>> # top-3 = [0.9, 0.7, 0.3] → 2 are positives → precision = 2/3 = 0.667
    """
    # Filter out samples with no valid candidates
    valid_candidates_mask = logits > float("-inf")
    num_valid_per_sample = valid_candidates_mask.sum(dim=1)
    sufficient_candidates = num_valid_per_sample >= k
    
    positives_per_sample = labels.sum(dim=1)
    valid_samples = (positives_per_sample > 0) & sufficient_candidates
    
    if valid_samples.sum() == 0:
        return 0.0, 0
    
    # Get top-K predictions
    k_actual = min(k, logits.size(1))
    _, top_k_indices = torch.topk(logits, k=k_actual, dim=1)
    
    # Get labels for top-K items
    top_k_labels = torch.gather(labels, 1, top_k_indices)
    hits_per_sample = top_k_labels.sum(dim=1)
    
    # Precision = hits / K
    precision_per_sample = hits_per_sample / k_actual
    
    # Average over valid samples
    total_precision = (precision_per_sample * valid_samples.float()).sum().item()
    num_valid = valid_samples.sum().item()
    
    return total_precision / num_valid if num_valid > 0 else 0.0, num_valid


def compute_batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Compute multiple metrics for a batch (used during training validation).
    
    This is a convenience function that computes Recall@K and Precision@K
    for multiple K values.
    
    Based on pb_insp_tt's metric computation approach.
    
    Args:
        logits: Scores for items [B, N]
        labels: Binary labels [B, N]
        k_values: List of K values for Recall@K and Precision@K
    
    Returns:
        Dictionary with all metrics:
        {
            "Recall@10": 0.75,
            "Precision@10": 0.60,
            "num_samples": 32,
        }
    """
    metrics = {}
    
    # Recall@K for each K
    num_samples = 0
    for k in k_values:
        recall, n = compute_recall_at_k(logits, labels, k=k)
        metrics[f"Recall@{k}"] = recall
        num_samples = max(num_samples, n)
    
    # Precision@K for each K
    for k in k_values:
        precision, _ = compute_precision_at_k(logits, labels, k=k)
        metrics[f"Precision@{k}"] = precision
    
    metrics["num_samples"] = num_samples
    
    return metrics


# def compute_full_metrics(
#     ranked_items: List[str],
#     ground_truth: List[str],
#     k_values_recall: List[int] = [10, 50, 100, 500],
#     k_values_precision: List[int] = [1, 5, 10, 20],
# ) -> Dict[str, Optional[float]]:
#     """
#     Compute full metrics for a single sample (used in eval.py).
    
#     This is used for comprehensive evaluation where ALL candidates are ranked
#     (not just sampled negatives).
    
#     Based on pb_insp_tt's get_sample_metrics function.
    
#     Args:
#         ranked_items: List of item IDs ranked by score (highest first)
#         ground_truth: List of ground truth positive item IDs
#         k_values_recall: K values for Recall@K computation
#         k_values_precision: K values for Precision@K computation
    
#     Returns:
#         Dictionary with metrics:
#         {
#             "Recall@10": 0.8,
#             "Recall@50": 0.95,
#             "Precision@1": 1.0,
#             "Precision@5": 0.6,
#             ...
#         }
        
#     Example:
#         >>> ranked = ["item_A", "item_B", "item_C", "item_D", "item_E"]
#         >>> ground_truth = ["item_A", "item_C", "item_F"]
#         >>> metrics = compute_full_metrics(ranked, ground_truth, k_values_recall=[3, 5])
#         >>> # top-3: [A, B, C] → hits = {A, C} → recall = 2/3 = 0.667
#         >>> # top-5: [A, B, C, D, E] → hits = {A, C} → recall = 2/3 = 0.667
#         >>> # precision@3 = 2/3 = 0.667
#     """
#     # Convert to sets for efficient intersection
#     ground_truth_set = set([str(item) for item in ground_truth])
#     ranked_items_str = [str(item) for item in ranked_items]
    
#     metrics = {}
    
#     # Recall@K
#     for k in k_values_recall:
#         top_k_candidates = set(ranked_items_str[:k])
#         intersection = ground_truth_set.intersection(top_k_candidates)
#         recall = (
#             len(intersection) / len(ground_truth_set)
#             if len(ground_truth_set) > 0
#             else None
#         )
#         metrics[f"Recall@{k}"] = recall
    
#     # Precision@K
#     for k in k_values_precision:
#         top_k_candidates = set(ranked_items_str[:k])
#         intersection = ground_truth_set.intersection(top_k_candidates)
#         precision = len(intersection) / k if k > 0 else None
#         metrics[f"Precision@{k}"] = precision
    
#     return metrics


def compute_mrr(
    ranked_items: List[str],
    ground_truth: List[str],
) -> Optional[float]:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant item)
    
    Useful for scenarios where you care about the rank of the first correct prediction.
    
    Args:
        ranked_items: List of item IDs ranked by score
        ground_truth: List of ground truth positive item IDs
    
    Returns:
        MRR score, or None if no relevant items found
        
    Example:
        >>> ranked = ["item_B", "item_A", "item_C"]
        >>> ground_truth = ["item_A", "item_D"]
        >>> mrr = compute_mrr(ranked, ground_truth)
        >>> # First relevant item is "item_A" at rank 2 → MRR = 1/2 = 0.5
    """
    ground_truth_set = set([str(item) for item in ground_truth])
    
    for rank, item in enumerate(ranked_items, start=1):
        if str(item) in ground_truth_set:
            return 1.0 / rank
    
    return None  # No relevant items found


def compute_ndcg(
    ranked_items: List[str],
    ground_truth: List[str],
    k: int = 10,
) -> Optional[float]:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG measures ranking quality with position-based discounting:
    - Items at higher positions contribute more to the score
    - Perfect ranking gets NDCG = 1.0
    
    Args:
        ranked_items: List of item IDs ranked by score
        ground_truth: List of ground truth positive item IDs
        k: Number of top items to consider
    
    Returns:
        NDCG@K score, or None if no relevant items
        
    Example:
        >>> ranked = ["item_A", "item_B", "item_C"]  # A is relevant
        >>> ground_truth = ["item_A", "item_D"]
        >>> ndcg = compute_ndcg(ranked, ground_truth, k=3)
        >>> # DCG = 1/log2(1+1) = 1.0 (item_A at position 1)
        >>> # IDCG = 1/log2(2) = 1.0 (perfect: relevant items first)
        >>> # NDCG = DCG / IDCG = 1.0
    """
    ground_truth_set = set([str(item) for item in ground_truth])
    
    if len(ground_truth_set) == 0:
        return None
    
    # Compute DCG@K
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if str(item) in ground_truth_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # Compute IDCG@K (ideal DCG - all relevant items at top)
    idcg = 0.0
    for i in range(1, min(len(ground_truth_set), k) + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    if idcg == 0:
        return None
    
    return dcg / idcg


# Convenience functions for logging

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary for logging.
    
    Args:
        metrics: Dictionary of metric_name -> value
        prefix: Optional prefix for metric names (e.g., "Val/")
    
    Returns:
        Formatted string for logging
        
    Example:
        >>> metrics = {"Recall@10": 0.756, "AUC": 0.823}
        >>> print(format_metrics(metrics, prefix="Val/"))
        Val/Recall@10: 0.7560, Val/AUC: 0.8230
    """
    formatted = []
    for name, value in metrics.items():
        if value is not None:
            full_name = f"{prefix}{name}" if prefix else name
            formatted.append(f"{full_name}: {value:.4f}")
    return ", ".join(formatted)


def aggregate_sample_metrics(
    sample_metrics_list: List[Dict[str, Optional[float]]]
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple samples (for eval.py).
    
    Computes mean of each metric across all samples, skipping None values.
    
    Args:
        sample_metrics_list: List of metric dictionaries from individual samples
    
    Returns:
        Aggregated metrics with mean values
        
    Example:
        >>> samples = [
        ...     {"Recall@10": 0.8, "Precision@10": 0.6},
        ...     {"Recall@10": 0.6, "Precision@10": None},
        ... ]
        >>> agg = aggregate_sample_metrics(samples)
        >>> # Recall@10: mean([0.8, 0.6]) = 0.7
        >>> # Precision@10: mean([0.6]) = 0.6 (skip None)
    """
    if not sample_metrics_list:
        return {}
    
    # Collect all metric names
    all_keys = set()
    for metrics in sample_metrics_list:
        all_keys.update(metrics.keys())
    
    # Compute mean for each metric
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in sample_metrics_list if key in m and m[key] is not None]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated
