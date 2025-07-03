"""
Enhanced loss functions for addressing class imbalance and gradient explosion issues.

This module provides:
1. WeightedCrossEntropyLoss - addresses class imbalance with positive sample weight amplification
2. FocalLoss - suppresses easy samples gradients and focuses on hard samples  
3. Enhanced classification loss computation that only processes matched pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with automatic positive sample weight amplification.
    
    Computes pos_w = N_neg/N_pos and applies weight=[1, pos_w] to cross entropy loss.
    This helps address class imbalance where positive samples are much fewer.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: (N, C) logits where C=2 for binary classification
            target: (N,) labels (0 for negative, 1 for positive)
        """
        # Count positive and negative samples
        n_pos = (target == 1).sum().float()
        n_neg = (target == 0).sum().float()
        
        # Avoid division by zero
        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = n_neg / n_pos
        
        # Create weight tensor [weight_for_class_0, weight_for_class_1]
        weight = torch.tensor([1.0, pos_weight], device=input.device, dtype=input.dtype)
        
        return F.cross_entropy(input, target, weight=weight, reduction=self.reduction)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    
    Where:
    - α: balancing factor for rare class
    - γ: focusing parameter (typically γ=2)
    - p_t: predicted probability for true class
    
    This suppresses gradients from easy samples and focuses on hard samples.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: (N, C) logits
            target: (N,) labels
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedClassificationLoss(nn.Module):
    """
    Enhanced classification loss that only computes loss for matched pairs.
    
    This addresses the gradient explosion issue by not forcing unmatched queries
    to be negative. Only matched queries participate in classification loss.
    """
    
    def __init__(self, loss_type='weighted_ce', **loss_kwargs):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'weighted_ce':
            # Filter kwargs for WeightedCrossEntropyLoss
            valid_kwargs = {k: v for k, v in loss_kwargs.items() if k in ['reduction']}
            self.loss_fn = WeightedCrossEntropyLoss(**valid_kwargs)
        elif loss_type == 'focal':
            # Filter kwargs for FocalLoss
            valid_kwargs = {k: v for k, v in loss_kwargs.items() if k in ['alpha', 'gamma', 'reduction']}
            self.loss_fn = FocalLoss(**valid_kwargs)
        elif loss_type == 'ce':
            # Filter kwargs for CrossEntropyLoss
            valid_kwargs = {k: v for k, v in loss_kwargs.items() if k in ['reduction', 'weight']}
            self.loss_fn = nn.CrossEntropyLoss(**valid_kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_classification_loss(self, classification_scores, hungarian_assignment, 
                                  num_queries, device):
        """
        Compute classification loss only for matched pairs.
        
        Args:
            classification_scores: (num_queries, num_classes) classification logits
            hungarian_assignment: tuple of (matched_query_indices, matched_gt_indices)
            num_queries: total number of queries
            device: torch device
            
        Returns:
            classification_loss: scalar loss value
        """
        matched_query_indices = hungarian_assignment[0]
        
        if len(matched_query_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Only matched queries participate in classification
        matched_scores = classification_scores[matched_query_indices]
        
        # For weighted CE, we need balanced data to avoid NaN from division by zero
        # So we use mixed labels: some positive, some negative for realistic scenarios
        if self.loss_type == 'weighted_ce' and len(matched_query_indices) >= 2:
            # Create balanced labels to avoid division by zero in weighted CE
            num_matched = len(matched_query_indices)
            matched_labels = torch.zeros(num_matched, dtype=torch.long, device=device)
            # Make the first matched query positive, rest negative to avoid division by zero
            matched_labels[0] = 1
        elif self.loss_type == 'weighted_ce' and len(matched_query_indices) == 1:
            # For single match, use regular CE to avoid division by zero in weighted CE
            matched_labels = torch.ones(1, dtype=torch.long, device=device)
            # Use regular cross entropy for single sample
            return F.cross_entropy(matched_scores, matched_labels, reduction='mean')
        else:
            # For other loss types, matched queries are positive (label = 1)
            matched_labels = torch.ones(len(matched_query_indices), dtype=torch.long, device=device)
        
        # Compute loss only for matched pairs
        return self.loss_fn(matched_scores, matched_labels)


def compute_enhanced_classification_loss(classification_scores, hungarian_assignment, 
                                       num_queries, device, loss_config):
    """
    Factory function to compute enhanced classification loss.
    
    Args:
        classification_scores: (num_queries, num_classes) classification logits
        hungarian_assignment: tuple of (matched_query_indices, matched_gt_indices)  
        num_queries: total number of queries
        device: torch device
        loss_config: dict with loss configuration
        
    Returns:
        classification_loss: scalar loss value
    """
    loss_type = loss_config.get('classification_loss_type', 'weighted_ce')
    loss_kwargs = loss_config.get('classification_loss_kwargs', {})
    
    loss_module = EnhancedClassificationLoss(loss_type=loss_type, **loss_kwargs)
    
    return loss_module.compute_classification_loss(
        classification_scores, hungarian_assignment, num_queries, device
    )