
# ============================================================================
# LOSS FUNCTIONS (Equation 8-14)
# ============================================================================

class ComprehensiveLoss(nn.Module):
    """
    Comprehensive loss function implementing Equation 8 from the paper.
    L_total = L_seg + λ1*L_consistency + λ2*L_uncertainty + λ3*L_diversity + λ4*L_sparsity + λ5*L_boundary
    """
    def __init__(self,
                 num_classes: int = 6,
                 lambda_consistency: float = 0.1,
                 lambda_uncertainty: float = 0.05,
                 lambda_diversity: float = 0.1,
                 lambda_sparsity: float = 0.01,
                 lambda_boundary: float = 0.2,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super(ComprehensiveLoss, self).__init__()
        
        self.num_classes = num_classes
        self.lambda_consistency = lambda_consistency
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_diversity = lambda_diversity
        self.lambda_sparsity = lambda_sparsity
        self.lambda_boundary = lambda_boundary
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss for segmentation (Eq. 8)
        """
        # pred: [B, C, H, W], target: [B, H, W]
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def consistency_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Consistency loss between ensemble members (Eq. 9)
        Encourages agreement between different encoders
        """
        # attention_weights: [B, M]
        M = attention_weights.shape[1]
        
        # Compute pairwise KL divergence
        kl_div = 0.0
        count = 0
        for i in range(M):
            for j in range(i+1, M):
                p_i = attention_weights[:, i:i+1]
                p_j = attention_weights[:, j:j+1]
                
                # KL(p_i || p_j)
                kl = p_i * torch.log((p_i + 1e-8) / (p_j + 1e-8))
                kl_div += kl.mean()
                count += 1
        
        return kl_div / max(count, 1)
    
    def uncertainty_loss(self, 
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Uncertainty calibration loss (Eq. 10)
        """
        # Compute prediction error
        pred_class = torch.argmax(pred, dim=1)
        error = (pred_class != target).float()
        
        # Average uncertainty across scales
        avg_uncertainty = uncertainty.mean(dim=1)  # [B, H, W]
        
        # Match spatial dimensions
        if avg_uncertainty.shape != error.shape:
            avg_uncertainty = F.interpolate(
                avg_uncertainty.unsqueeze(1),
                size=error.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # MSE between uncertainty and actual error
        uncertainty_loss = F.mse_loss(avg_uncertainty, error)
        
        return uncertainty_loss
    
    def diversity_loss(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Diversity loss to encourage feature diversity (Eq. 11)
        Promotes orthogonality between different encoder features
        """
        # Flatten and normalize features
        flattened = []
        for feat in encoder_features:
            feat_flat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            feat_norm = F.normalize(feat_flat, p=2, dim=1)
            flattened.append(feat_norm)
        
        # Compute pairwise cosine similarity
        M = len(flattened)
        diversity = 0.0
        count = 0
        
        for i in range(M):
            for j in range(i+1, M):
                cos_sim = (flattened[i] * flattened[j]).sum(dim=1).mean()
                diversity += cos_sim
                count += 1
        
        # Negative diversity (want to minimize similarity)
        return -diversity / max(count, 1)
    
    def sparsity_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Sparsity regularization on attention weights (Eq. 12)
        Encourages sparse attention distribution
        """
        # L1 regularization on attention weights
        return attention_weights.abs().mean()
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Boundary-aware loss (Eq. 13)
        Emphasizes accuracy at object boundaries
        """
        # Compute boundary mask using Sobel operator
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # Compute boundaries for each class
        boundaries = []
        for c in range(self.num_classes):
            grad_x = F.conv2d(target_one_hot[:, c:c+1], sobel_x, padding=1)
            grad_y = F.conv2d(target_one_hot[:, c:c+1], sobel_y, padding=1)
            boundary = torch.sqrt(grad_x**2 + grad_y**2)
            boundaries.append(boundary)
        
        boundary_mask = torch.cat(boundaries, dim=1)  # [B, C, H, W]
        boundary_mask = (boundary_mask > 0.1).float()
        
        # Weighted CE loss at boundaries
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        boundary_weight = boundary_mask.sum(dim=1) + 1.0  # [B, H, W]
        weighted_loss = (ce_loss * boundary_weight).mean()
        
        return weighted_loss
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                attention_weights: torch.Tensor,
                encoder_features: Optional[List[torch.Tensor]] = None,
                uncertainty: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss (Eq. 8)
        
        Args:
            pred: Predicted segmentation [B, C, H, W]
            target: Ground truth [B, H, W]
            attention_weights: Encoder attention weights [B, M]
            encoder_features: List of encoder feature maps (for diversity loss)
            uncertainty: Scale uncertainty maps [B, num_scales, H, W]
            
        Returns:
            Dictionary of losses
        """
        # Segmentation loss (focal loss)
        loss_seg = self.focal_loss(pred, target)
        
        # Consistency loss
        loss_consistency = self.consistency_loss(attention_weights)
        
        # Uncertainty loss
        if uncertainty is not None:
            loss_uncertainty = self.uncertainty_loss(pred, target, uncertainty)
        else:
            loss_uncertainty = torch.tensor(0.0, device=pred.device)
        
        # Diversity loss
        if encoder_features is not None:
            loss_diversity = self.diversity_loss(encoder_features)
        else:
            loss_diversity = torch.tensor(0.0, device=pred.device)
        
        # Sparsity loss
        loss_sparsity = self.sparsity_loss(attention_weights)
        
        # Boundary loss
        loss_boundary = self.boundary_loss(pred, target)
        
        # Total loss (Eq. 8)
        total_loss = (loss_seg +
                     self.lambda_consistency * loss_consistency +
                     self.lambda_uncertainty * loss_uncertainty +
                     self.lambda_diversity * loss_diversity +
                     self.lambda_sparsity * loss_sparsity +
                     self.lambda_boundary * loss_boundary)
        
        return {
            'total': total_loss,
            'segmentation': loss_seg,
            'consistency': loss_consistency,
            'uncertainty': loss_uncertainty,
            'diversity': loss_diversity,
            'sparsity': loss_sparsity,
            'boundary': loss_boundary
        }


