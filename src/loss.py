import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Dict, Tuple, Set, Any


class PurchasedNegativeSampler:
    """
    Reusable negative sampler for sampling from purchased negatives (label=0 items).
    
    Supports multiple sampling strategies:
    - "random": Pure random sampling
    - "hard": Pure hard negative mining (top-K highest scoring)
    - "frequency": Sample most frequently purchased items (requires membership data)
    - "recency": Sample most recently purchased items (requires membership data)
    - "mixed": Custom mix of strategies (specify via strategy_weights)
    
    Returns INDICES of sampled negatives, making it reusable across different loss functions.
    """
    
    def __init__(
        self,
        sampling_strategy: str = "random",
        frequency_window_days: int = 364,
        strategy_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            sampling_strategy: Strategy for negative sampling
            frequency_window_days: Time window for frequency sampling (e.g., 28, 91, 182 days)
            strategy_weights: For "mixed" strategy or custom weights, e.g.:
                {"hard": 0.4, "frequency": 0.3, "recency": 0.2, "random": 0.1}
        """
        self.sampling_strategy = sampling_strategy
        self.frequency_window_days = frequency_window_days
        
        # Strategy weights - controls sampling behavior
        if strategy_weights is not None:
            # User-provided weights
            total_weight = sum(strategy_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(
                    f"strategy_weights sum to {total_weight:.3f}, normalizing to 1.0"
                )
                self.strategy_weights = {
                    k: v / total_weight for k, v in strategy_weights.items()
                }
            else:
                self.strategy_weights = strategy_weights
        else:
            # Auto-set weights based on sampling_strategy
            if sampling_strategy == "hard":
                self.strategy_weights = {"hard": 1.0}
            elif sampling_strategy == "frequency":
                self.strategy_weights = {"frequency": 1.0}
            elif sampling_strategy == "recency":
                self.strategy_weights = {"recency": 1.0}
            elif sampling_strategy == "mixed":
                # Default balanced mix
                self.strategy_weights = {
                    "hard": 0.4,
                    "frequency": 0.3,
                    "recency": 0.2,
                    "random": 0.1,
                }
                logger.info(
                    f"No strategy_weights provided for mixed sampling. Using default: {self.strategy_weights}"
                )
            else:
                # Random or unknown: pure random sampling
                self.strategy_weights = {"random": 1.0}
    
    def sample(
        self,
        num_samples: int,
        neg_seq_positions: torch.Tensor,           # [N] - sequence positions of negatives
        neg_scores: Optional[torch.Tensor] = None,  # [N] - scores (required for "hard" strategy)
        membership: Optional[torch.Tensor] = None,  # [S, 364] - membership history (required for frequency/recency)
    ) -> torch.Tensor:
        """
        Sample negative indices from purchased negatives.
        
        Args:
            num_samples: Number of negatives to sample
            neg_seq_positions: Sequence positions of purchased negatives [N]
            neg_scores: Scores for hard negative mining (optional, required for "hard" strategy)
            membership: Membership history [S, 364] (optional, required for frequency/recency)
        
        Returns:
            Tensor of sampled indices [num_sampled] into neg_seq_positions/neg_scores
        """
        num_available = neg_seq_positions.size(0)
        device = neg_seq_positions.device
        
        # Early return if we need all negatives or no sampling needed
        if num_samples >= num_available:
            logger.debug(
                f"Requested {num_samples} negative samples but only {num_available} available. "
                f"Using all {num_available} negatives."
            )
            return torch.arange(num_available, device=device, dtype=torch.long)
        
        # Safety check: if strategy_weights is None, use pure random
        if self.strategy_weights is None:
            logger.warning(
                "strategy_weights is None. Using pure random sampling."
            )
            return torch.randperm(num_available, device=device)[:num_samples]
        
        sampled_indices: Set[int] = set()
        
        # Apply each strategy according to weights
        strategy_list = list(self.strategy_weights.items())
        for i, (strategy_name, weight) in enumerate(strategy_list):
            if weight <= 0:
                continue
            
            # For the last strategy, allocate all remaining samples to avoid rounding errors
            if i == len(strategy_list) - 1:
                strategy_samples = num_samples - len(sampled_indices)
            else:
                strategy_samples = int(num_samples * weight)
            
            if strategy_samples <= 0:
                continue
            
            if strategy_name == "hard":
                if neg_scores is None:
                    logger.warning(
                        f"Strategy 'hard' (weight={weight:.2f}) requires neg_scores but it's unavailable. "
                        f"Skipping hard sampling."
                    )
                    continue
                
                # Select top-K highest scoring
                num_to_select = min(strategy_samples, num_available)
                if num_to_select > 0:
                    _, top_indices = torch.topk(neg_scores, k=num_to_select, largest=True)
                    for idx in top_indices.tolist():
                        if idx not in sampled_indices:
                            sampled_indices.add(idx)
            
            elif strategy_name == "frequency":
                if membership is None:
                    logger.warning(
                        f"Strategy 'frequency' (weight={weight:.2f}) requires membership data but it's unavailable. "
                        f"Skipping frequency sampling for this batch."
                    )
                else:
                    # Frequency-based: sort by purchase count
                    # NOTE: membership is ordered reversely - index 0 = yesterday, index 363 = 364 days ago
                    neg_membership = membership[neg_seq_positions]
                    if self.frequency_window_days < 364:
                        # Take first N days (most recent)
                        frequency = neg_membership[:, :self.frequency_window_days].sum(dim=1)
                    else:
                        frequency = neg_membership.sum(dim=1)
                    
                    # Sort by frequency, excluding already sampled
                    sorted_by_freq = torch.argsort(frequency, descending=True)
                    count = 0
                    for idx in sorted_by_freq.tolist():
                        if idx not in sampled_indices:
                            sampled_indices.add(idx)
                            count += 1
                            if count >= strategy_samples:
                                break
            
            elif strategy_name == "recency":
                if membership is None:
                    logger.warning(
                        f"Strategy 'recency' (weight={weight:.2f}) requires membership data but it's unavailable. "
                        f"Skipping recency sampling for this batch."
                    )
                else:
                    # Recency-based: sort by first occurrence (most recent)
                    # NOTE: membership is ordered reversely - index 0 = yesterday, index 363 = 364 days ago
                    # Vectorized version: neg_membership: [N, 364] with 0/1
                    neg_membership = membership[neg_seq_positions]
                    has_any = neg_membership.any(dim=1)  # [N] - True if any purchase history
                    # argmax gives first index of 1 (most recent), or 0 if all zeros
                    # Use torch.where to set 999 for items with no purchase history
                    first_occurrence = torch.where(
                        has_any,
                        neg_membership.float().argmax(dim=1),
                        torch.tensor(999, device=device, dtype=torch.long)
                    )
                    
                    # Sort by first occurrence in ascending order (lower index = more recent)
                    sorted_by_recency = torch.argsort(first_occurrence, descending=False)
                    count = 0
                    for idx in sorted_by_recency.tolist():
                        if idx not in sampled_indices:
                            sampled_indices.add(idx)
                            count += 1
                            if count >= strategy_samples:
                                break
            
            elif strategy_name == "random":
                # Random sampling from remaining available indices
                remaining = list(set(range(num_available)) - sampled_indices)
                if len(remaining) > 0:
                    num_to_select = min(strategy_samples, len(remaining))
                    selected = torch.tensor(remaining, device=device)[
                        torch.randperm(len(remaining), device=device)[:num_to_select]
                    ]
                    for idx in selected.tolist():
                        sampled_indices.add(idx)
            else:
                logger.warning(f"Unknown strategy in sampling: '{strategy_name}'")
        
        # Fill remainder with random as a safety net (should rarely trigger now)
        if len(sampled_indices) < num_samples:
            remaining = list(set(range(num_available)) - sampled_indices)
            if remaining:
                need = num_samples - len(sampled_indices)
                extra = torch.tensor(remaining, device=device)[
                    torch.randperm(len(remaining), device=device)[:min(need, len(remaining))]
                ]
                sampled_indices.update(extra.tolist())
        
        # Warn if we STILL didn't get the expected number of samples (should basically never happen)
        if len(sampled_indices) < num_samples:
            logger.warning(
                f"Sampled only {len(sampled_indices)} negatives (requested {num_samples}). "
                f"This should only happen if num_available < num_samples."
            )
        
        # Convert to tensor and return indices
        if len(sampled_indices) > 0:
            return torch.tensor(sorted(sampled_indices), device=device, dtype=torch.long)
        else:
            return torch.arange(num_available, device=device, dtype=torch.long)


class RankerLoss(nn.Module):
    """
    Flexible margin-based ranking loss for repurchase prediction.
    
    Features:
    - Pairwise margin ranking loss
    - Negative sampling FROM purchased items only (label=0 items)
    - Unified implementation for all sampling strategies via strategy_weights
    - Sampling ratio: num_negatives = num_positives * neg_sample_ratio
    
    Sampling Strategies (controlled by sampling_strategy parameter):
    - "random": Pure random sampling
    - "hard": Pure hard negative mining (top-K highest scoring)
    - "frequency": Sample most frequently purchased items (requires membership data)
    - "recency": Sample most recently purchased items (requires membership data)
    - "mixed": Custom mix of strategies (specify via strategy_weights)
    
    Example:
    - 10 positives, 490 purchased negatives, ratio=10 → sample 100 from 490 negatives
    - Pure frequency: {"frequency": 1.0} → top 100 by frequency
    - Mixed: {"hard": 0.5, "frequency": 0.3, "random": 0.2} → 50 hard + 30 frequency + 20 random
    """

    def __init__(
        self,
        margin=1.0,
        neg_sample_ratio=5,
        # Advanced sampling strategies
        sampling_strategy="random",  # "random", "frequency", "recency", "hard", "mixed"
        frequency_window_days=28,     # Consider last N days for frequency
        strategy_weights=None,        # For mixed strategy: {"hard": 0.4, "frequency": 0.3, ...}
    ):
        """
        Args:
            margin: Margin for ranking loss. Higher margin means larger gap between pos/neg scores.
            neg_sample_ratio: Number of negatives to sample per positive.
                Example: 10 positives * ratio=5 = 50 negatives sampled
            sampling_strategy: Strategy for negative sampling:
                - "random": Pure random sampling from purchased negatives
                - "hard": Pure hard negative mining (top-K highest scoring)
                - "frequency": Pure frequency-based (most frequently purchased in recent window)
                - "recency": Pure recency-based (most recently purchased)
                - "mixed": Combine multiple strategies with custom weights
            frequency_window_days: Time window for frequency sampling (e.g., 28, 91, 182 days)
                - For frequency: counts purchases in FIRST N days (most recent)
            strategy_weights: For "mixed" strategy only. Dict of strategy weights, e.g.:
                {"hard": 0.4, "frequency": 0.3, "recency": 0.2, "random": 0.1}
                Weights should sum to ~1.0 (will be normalized if not)
                Can also override default behavior for any strategy, e.g.:
                sampling_strategy="hard", strategy_weights={"hard": 0.7, "random": 0.3}
        """
        super().__init__()
        self.margin = margin
        self.neg_sample_ratio = neg_sample_ratio
        
        # Create reusable negative sampler
        self.negative_sampler = PurchasedNegativeSampler(
            sampling_strategy=sampling_strategy,
            frequency_window_days=frequency_window_days,
            strategy_weights=strategy_weights,
        )

    def forward(
        self,
        final_scores,    # [B, vocab_size] - model output scores for all items
        item_indices,    # [B, S] - vocabulary indices for purchased items
        labels,          # [B, S] - repurchase labels (1=will repurchase, 0=won't)
        attention_mask,  # [B, S] - True for valid positions
        membership=None, # [B, S, 364] - membership history (for frequency/recency sampling)
        **kwargs,
    ):
        """
        Compute pairwise margin ranking loss with negative sampling from purchased items.
        
        Positives: Items with label=1 (will repurchase)
        Negatives: Sampled from items with label=0 (purchased but won't repurchase)
                   Number sampled = num_positives * neg_sample_ratio
        
        Loss: For each (positive, negative) pair:
              max(0, margin - score_positive + score_negative)
        
        Args:
            final_scores: [B, vocab_size] - model output scores for all items in vocabulary
            item_indices: [B, S] - vocabulary indices (NOT raw item_ids!) for purchased items
            labels: [B, S] - repurchase labels (1=will repurchase, 0=won't)
            attention_mask: [B, S] - True for valid positions
            membership: [B, S, 364] - membership history (for frequency/recency sampling)
        
        Returns:
            dict with:
                - loss: total ranking loss
                - logits: combined scores [B, total_items] for metrics
                - labels: binary labels [B, total_items] for metrics
                - num_positives: number of positive items
                - num_negatives: number of negative items (sampled)
                - num_hard_negatives: number of hard negatives (if using hard neg mining)
        """
        device = final_scores.device
        B, S = item_indices.shape
        vocab_size = final_scores.size(1)

        # Extract scores for purchased items
        # item_indices: [B, S], final_scores: [B, vocab_size]
        # purchased_scores: [B, S]
        purchased_scores = torch.gather(final_scores, 1, item_indices)

        # Create masks for positives and negatives
        valid_mask = attention_mask  # True for valid positions (not padding)
        positive_mask = (labels == 1) & valid_mask  # Will repurchase
        negative_mask_purchased = (labels == 0) & valid_mask  # Purchased but won't repurchase

        # Collect positive and negative scores per batch
        positive_scores_list = []
        negative_scores_list = []

        max_positives = 0
        max_negatives = 0

        for b in range(B):
            # Positive scores for this sample
            pos_indices = positive_mask[b]
            pos_scores = purchased_scores[b, pos_indices]  # [N_pos]
            num_pos = pos_scores.size(0)
            max_positives = max(max_positives, num_pos)

            # Negative scores from purchased items (label=0)
            neg_indices_purchased = negative_mask_purchased[b]
            neg_scores_purchased = purchased_scores[b, neg_indices_purchased]  # [N_neg_purchased]
            neg_seq_positions = torch.where(neg_indices_purchased)[0]  # Sequence positions of negatives
            num_available_negs = neg_scores_purchased.size(0)

            # Get membership data for this sample if available
            sample_membership = membership[b] if membership is not None else None

            # Determine how many negatives to sample
            num_negs_to_sample = min(
                num_pos * self.neg_sample_ratio if num_pos > 0 else 0,
                num_available_negs
            )

            # Apply sampling strategy to purchased negatives only
            if num_negs_to_sample == 0 or num_available_negs == 0:
                # No negatives to sample
                neg_scores_all = neg_scores_purchased
            elif num_negs_to_sample >= num_available_negs:
                # Use all available purchased negatives
                neg_scores_all = neg_scores_purchased
            else:
                # Use negative sampler to get indices, then select scores
                sampled_indices = self.negative_sampler.sample(
                    num_samples=num_negs_to_sample,
                    neg_seq_positions=neg_seq_positions,
                    neg_scores=neg_scores_purchased,
                    membership=sample_membership,
                )
                neg_scores_all = neg_scores_purchased[sampled_indices]

            num_neg = neg_scores_all.size(0)
            max_negatives = max(max_negatives, num_neg)

            positive_scores_list.append(pos_scores)
            negative_scores_list.append(neg_scores_all)

        # Check if we have valid samples
        if max_positives == 0 or max_negatives == 0:
            logger.debug("No valid positive or negative samples found for ranking loss")
            # Return a dummy loss with gradient tracking to avoid breaking backward pass
            dummy_loss = torch.zeros(1, device=device, requires_grad=True)
            return {
                "loss": dummy_loss,
                "logits": None,
                "labels": None,
                "num_positives": 0,
                "num_negatives": 0,
                "num_hard_negatives": 0,
            }

        # Pad to create tensors
        positive_scores_padded = torch.full(
            (B, max_positives), float("-inf"), device=device
        )
        negative_scores_padded = torch.full(
            (B, max_negatives), float("-inf"), device=device
        )
        positive_mask_padded = torch.ones(
            (B, max_positives), dtype=torch.bool, device=device
        )  # True = padding
        negative_mask_padded = torch.ones(
            (B, max_negatives), dtype=torch.bool, device=device
        )

        for b in range(B):
            num_pos = positive_scores_list[b].size(0)
            num_neg = negative_scores_list[b].size(0)

            if num_pos > 0:
                positive_scores_padded[b, :num_pos] = positive_scores_list[b]
                positive_mask_padded[b, :num_pos] = False

            if num_neg > 0:
                negative_scores_padded[b, :num_neg] = negative_scores_list[b]
                negative_mask_padded[b, :num_neg] = False

        # Compute pairwise margin ranking loss
        loss_ranking = self._pairwise_margin_loss(
            positive_scores=positive_scores_padded,
            negative_scores=negative_scores_padded,
            positive_mask=positive_mask_padded,
            negative_mask=negative_mask_padded,
                )

        if loss_ranking is None:
            loss_ranking = torch.tensor(0.0, device=device)

        # Prepare outputs for metrics
        logits, metric_labels = self._prepare_metric_outputs(
            positive_scores=positive_scores_padded,
            negative_scores=negative_scores_padded,
            positive_mask=positive_mask_padded,
            negative_mask=negative_mask_padded,
        )

        # Count actual positives and negatives (not padding)
        num_positives = (~positive_mask_padded).sum().item()
        num_negatives = (~negative_mask_padded).sum().item()

        return {
            "loss": loss_ranking,
            "logits": logits,
            "labels": metric_labels,
            "num_positives": num_positives,
            "num_negatives": num_negatives,
        }

    def _pairwise_margin_loss(
        self,
        positive_scores,  # [B, T] - scores for positive items
        negative_scores,  # [B, N] - scores for negative items
        positive_mask,    # [B, T] - True for padding
        negative_mask,    # [B, N] - True for padding
    ):
        """
        Compute pairwise margin ranking loss.
        
        For each (positive, negative) pair:
            loss = max(0, margin - score_positive + score_negative)
        
        The loss encourages positive scores to be higher than negative scores by at least 'margin'.
        """
        B, T = positive_scores.shape
        N = negative_scores.shape[1]

        # Check for valid samples
        has_valid_pos = ~positive_mask.all(dim=1)  # At least one valid positive
        has_valid_neg = ~negative_mask.all(dim=1)  # At least one valid negative
        valid_samples = has_valid_pos & has_valid_neg

        if not valid_samples.any():
            logger.warning("No valid samples for pairwise margin loss")
            return None

        # Expand for pairwise comparison: [B, T, N]
        pos_expanded = positive_scores.unsqueeze(2).expand(B, T, N)  # [B, T, N]
        neg_expanded = negative_scores.unsqueeze(1).expand(B, T, N)  # [B, T, N]

        # Pairwise margin loss
        # If pos_score < neg_score + margin, we incur a loss
        pairwise_loss = F.relu(self.margin - pos_expanded + neg_expanded)  # [B, T, N]

        # Create pairwise validity mask
        # Valid only if both positive and negative are not padding
        pos_valid = (~positive_mask).unsqueeze(2).expand(B, T, N)  # [B, T, N]
        neg_valid = (~negative_mask).unsqueeze(1).expand(B, T, N)  # [B, T, N]
        pairwise_valid = pos_valid & neg_valid  # [B, T, N]

        # Also mask out samples with no valid pos/neg
        sample_valid = valid_samples.view(B, 1, 1).expand(B, T, N)
        pairwise_valid = pairwise_valid & sample_valid

        # Average over valid pairs
        if pairwise_valid.sum() > 0:
            return pairwise_loss[pairwise_valid].mean()
        else:
            return torch.tensor(0.0, device=positive_scores.device)

    def _prepare_metric_outputs(
        self,
        positive_scores,  # [B, T]
        negative_scores,  # [B, N]
        positive_mask,    # [B, T] - True for padding
        negative_mask,    # [B, N] - True for padding
    ):
        """
        Prepare combined logits and labels for metric computation (AUC, accuracy, etc.).
        
        Returns:
            logits: [B, T+N] - combined scores with padding masked to -inf
            labels: [B, T+N] - 1.0 for valid positives, 0.0 for negatives and padding
        """
        num_positives = positive_scores.size(1)

        # Concatenate positive and negative scores
        logits = torch.cat([positive_scores, negative_scores], dim=-1)  # [B, T+N]

        # Create labels: 1 for valid positives, 0 for everything else
        labels = torch.zeros_like(logits)
        labels[:, :num_positives] = (~positive_mask).float()

        # Mask out padding positions (set to -inf for proper metric computation)
        full_mask = torch.cat([positive_mask, negative_mask], dim=-1)
        logits = logits.masked_fill(full_mask, float("-inf"))

        return logits, labels


class BCELoss(nn.Module):
    """
    BCE loss for repurchase prediction using the SAME negative sampling strategy
    as RankerLoss (sample only from purchased negatives: label=0 items).

    Design:
    - Positives: purchased items with label=1
    - Negatives: sampled from purchased items with label=0
      (num_neg = min(num_pos * neg_sample_ratio, available_neg))
    - Loss: BCEWithLogitsLoss on the concatenated (pos_logits, neg_logits)
    - Uses PurchasedNegativeSampler for consistent sampling strategies
    """

    def __init__(
        self,
        neg_sample_ratio: int = 5,
        sampling_strategy: str = "random",
        frequency_window_days: int = 28,
        strategy_weights: Optional[Dict[str, float]] = None,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            neg_sample_ratio: number of negatives to sample per positive
            sampling_strategy: "random" | "hard" | "frequency" | "recency" | "mixed"
            frequency_window_days: window for frequency/recency strategies
            strategy_weights: optional weights for "mixed" or custom strategy mix
            pos_weight: optional positive class weight for BCE (helps imbalance)
                        If set, applied inside BCEWithLogitsLoss.
            reduction: "mean" | "sum" (typically "mean")
        """
        super().__init__()
        self.neg_sample_ratio = neg_sample_ratio

        # Reuse the same sampler class you created
        self.negative_sampler = PurchasedNegativeSampler(
            sampling_strategy=sampling_strategy,
            frequency_window_days=frequency_window_days,
            strategy_weights=strategy_weights,
        )

        # BCE loss module (expects logits)
        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32)
        self._bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor,
            reduction=reduction,
        )

    def forward(
        self,
        final_scores: torch.Tensor,   # [B, vocab_size] logits for ALL items
        item_indices: torch.Tensor,   # [B, S] vocab indices for purchased items
        labels: torch.Tensor,         # [B, S] 1=will repurchase, 0=won't
        attention_mask: torch.Tensor, # [B, S] True for valid positions
        membership: Optional[torch.Tensor] = None,  # [B, S, 364]
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Returns:
            dict with:
              - loss
              - logits: [B, T+N] (padded with -inf)
              - labels: [B, T+N] (1 for positives, 0 for negatives, 0 for padding)
              - num_positives
              - num_negatives
        """
        device = final_scores.device
        B, S = item_indices.shape

        # Ensure pos_weight tensor is on correct device if used
        if self._bce.pos_weight is not None and self._bce.pos_weight.device != device:
            self._bce.pos_weight = self._bce.pos_weight.to(device)

        # [B, S] logits for purchased items only
        purchased_scores = torch.gather(final_scores, 1, item_indices)

        valid_mask = attention_mask
        positive_mask = (labels == 1) & valid_mask
        negative_mask_purchased = (labels == 0) & valid_mask

        pos_scores_list = []
        neg_scores_list = []
        max_pos = 0
        max_neg = 0

        for b in range(B):
            # Positives
            pos_idx = positive_mask[b]
            pos_scores = purchased_scores[b, pos_idx]  # [N_pos]
            num_pos = int(pos_scores.numel())
            max_pos = max(max_pos, num_pos)

            # Purchased negatives pool
            neg_idx = negative_mask_purchased[b]
            neg_scores_pool = purchased_scores[b, neg_idx]  # [N_neg_pool]
            neg_seq_positions = torch.where(neg_idx)[0]     # [N_neg_pool]
            num_avail = int(neg_scores_pool.numel())

            sample_membership = membership[b] if membership is not None else None

            # Decide how many negatives to sample
            num_to_sample = min(num_pos * self.neg_sample_ratio if num_pos > 0 else 0, num_avail)

            if num_to_sample <= 0 or num_avail == 0:
                neg_scores = neg_scores_pool
            elif num_to_sample >= num_avail:
                neg_scores = neg_scores_pool
            else:
                sampled = self.negative_sampler.sample(
                    num_samples=num_to_sample,
                    neg_seq_positions=neg_seq_positions,
                    neg_scores=neg_scores_pool,          # for "hard"
                    membership=sample_membership,        # for freq/recency
                )
                # sampled indexes into neg_scores_pool
                neg_scores = neg_scores_pool[sampled]

            num_neg = int(neg_scores.numel())
            max_neg = max(max_neg, num_neg)

            pos_scores_list.append(pos_scores)
            neg_scores_list.append(neg_scores)

        # If nothing usable, return dummy loss
        if max_pos == 0 or max_neg == 0:
            logger.debug("No valid positive or negative samples found for BCE loss")
            dummy_loss = torch.zeros(1, device=device, requires_grad=True)
            return {
                "loss": dummy_loss,
                "logits": None,
                "labels": None,
                "num_positives": 0,
                "num_negatives": 0,
            }

        # Pad for batching
        pos_padded = torch.full((B, max_pos), float("-inf"), device=device)
        neg_padded = torch.full((B, max_neg), float("-inf"), device=device)

        pos_pad_mask = torch.ones((B, max_pos), dtype=torch.bool, device=device)  # True=padding
        neg_pad_mask = torch.ones((B, max_neg), dtype=torch.bool, device=device)

        for b in range(B):
            np_ = pos_scores_list[b].numel()
            nn_ = neg_scores_list[b].numel()

            if np_ > 0:
                pos_padded[b, :np_] = pos_scores_list[b]
                pos_pad_mask[b, :np_] = False

            if nn_ > 0:
                neg_padded[b, :nn_] = neg_scores_list[b]
                neg_pad_mask[b, :nn_] = False

        # Build logits + targets
        logits, targets = self._prepare_metric_outputs(
            positive_scores=pos_padded,
            negative_scores=neg_padded,
            positive_mask=pos_pad_mask,
            negative_mask=neg_pad_mask,
        )

        # Compute BCE on valid positions only
        valid = logits != float("-inf")
        if valid.any():
            loss = self._bce(logits[valid], targets[valid])
        else:
            loss = torch.zeros(1, device=device, requires_grad=True)

        num_pos_total = int((~pos_pad_mask).sum().item())
        num_neg_total = int((~neg_pad_mask).sum().item())

        return {
            "loss": loss,
            "logits": logits,
            "labels": targets,
            "num_positives": num_pos_total,
            "num_negatives": num_neg_total,
        }

    @staticmethod
    def _prepare_metric_outputs(
        positive_scores: torch.Tensor,  # [B, T]
        negative_scores: torch.Tensor,  # [B, N]
        positive_mask: torch.Tensor,    # [B, T] True=padding
        negative_mask: torch.Tensor,    # [B, N] True=padding
    ):
        """
        Returns:
            logits: [B, T+N] with padding masked to -inf
            labels: [B, T+N] float targets (1 for valid positives, 0 for negatives/padding)
        """
        num_pos = positive_scores.size(1)

        logits = torch.cat([positive_scores, negative_scores], dim=-1)  # [B, T+N]
        labels = torch.zeros_like(logits)
        labels[:, :num_pos] = (~positive_mask).float()

        full_mask = torch.cat([positive_mask, negative_mask], dim=-1)
        logits = logits.masked_fill(full_mask, float("-inf"))

        return logits, labels
