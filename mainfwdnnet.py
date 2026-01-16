import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import List, Tuple, Dict, Optional

class TensorDotFusion(nn.Module):
    """
    we may appologize and request you that, for equation references may change based:
    on how we will be revising the paper for final production with IEEE,
    please be aware of this if you are refering to paper equations as guidance
    TensorDot Fusion Module implementing (see in main paper and supplimenatry materials at Equations 2-4) from the paper.
    Performs multilinear transformation to capture high-order interactions.
    """
    def __init__(self, 
                 input_channels: List[int], 
                 output_channels: int = 256,
                 reduction_ratio: int = 4):
        super(TensorDotFusion, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_encoders = len(input_channels)
        
        # Learnable core tensor G for high-order interactions (Eq. 3)
        # We use Tucker decomposition for efficiency
        self.core_rank = max(32, output_channels // reduction_ratio)
        
        # Project each encoder's features to common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, self.core_rank, 1, bias=False),
                nn.BatchNorm2d(self.core_rank),
                nn.ReLU(inplace=True)
            ) for in_ch in input_channels
        ])
        
        # Core tensor learnable weights
        self.core_weights = nn.Parameter(
            torch.randn(self.num_encoders, self.core_rank, output_channels) * 0.01
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_maps: List of [B, C_m, H, W] tensors from different encoders
        Returns:
            fused_features: [B, output_channels, H, W]
        """
        B, _, H, W = feature_maps[0].shape
        
        # Resize all features to same spatial dimensions
        target_size = (H, W)
        aligned_features = []
        for feat in feature_maps:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Project to common dimension (Eq. 3)
        projected = []
        for i, feat in enumerate(aligned_features):
            proj_feat = self.projections[i](feat)  # [B, core_rank, H, W]
            projected.append(proj_feat)
        
        # Stack and apply TensorDot operation
        stacked = torch.stack(projected, dim=1)  # [B, num_encoders, core_rank, H, W]
        
        # Multilinear transformation using core tensor
        # Reshape for batch matrix multiplication
        B, M, R, H, W = stacked.shape
        stacked_reshaped = stacked.permute(0, 3, 4, 1, 2).reshape(B*H*W, M, R)
        
        # Apply core tensor weights: [B*H*W, M, R] @ [M, R, output_channels]
        # Simplified: sum over encoders and project
        fused = torch.einsum('bmr,mro->bo', stacked_reshaped, self.core_weights)
        fused = fused.reshape(B, H, W, self.output_channels).permute(0, 3, 1, 2)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output


class ProbabilisticAttentionWeighting(nn.Module):
    """
    Probabilistic Attention Weighting with Variational Inference (see in a paper Eq. 4-5).
    Computes adaptive weights for each encoder based on input features.
    """
    def __init__(self, 
                 input_channels: List[int], 
                 hidden_dim: int = 128):
        super(ProbabilisticAttentionWeighting, self).__init__()
        
        self.num_encoders = len(input_channels)
        total_channels = sum(input_channels)
        
        # Global Average Pooling followed by MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.num_encoders)
        )
        
        # Variational parameters for uncertainty estimation (Eq. 5)
        self.log_var_mlp = nn.Sequential(
            nn.Linear(total_channels, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.num_encoders)
        )
        
    def forward(self, feature_maps: List[torch.Tensor], 
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_maps: List of feature tensors
            training: Whether in training mode for variational sampling
        Returns:
            attention_weights: [B, num_encoders, 1, 1]
            uncertainty: [B, num_encoders]
        """
        # Global average pooling for each encoder
        pooled_features = []
        for feat in feature_maps:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
            pooled_features.append(pooled)
        
        # Concatenate all pooled features
        concat_features = torch.cat(pooled_features, dim=1)  # [B, total_channels]
        
        # Compute attention logits
        attention_logits = self.attention_mlp(concat_features)  # [B, num_encoders]
        
        # Compute log variance for uncertainty
        log_var = self.log_var_mlp(concat_features)  # [B, num_encoders]
        
        # Variational sampling during training (Eq. 5)
        if training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            attention_logits = attention_logits + eps * std
        
        # Softmax to get probabilistic weights (Eq. 4)
        attention_weights = F.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1]
        
        uncertainty = torch.exp(log_var)
        
        return attention_weights, uncertainty


class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid with Uncertainty Quantification (Eq. 6-7).
    """
    def __init__(self, channels: int, num_scales: int = 4):
        super(MultiScaleFeaturePyramid, self).__init__()
        
        self.num_scales = num_scales
        
        # Scale-specific processing
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2**(i+1), dilation=2**(i+1), bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_scales)
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales, 1),
            nn.Softmax(dim=1)
        )
        
        # Uncertainty estimation per scale (Eq. 7)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            multi_scale_features: [B, C, H, W]
            scale_uncertainty: [B, num_scales, H, W]
        """
        B, C, H, W = x.shape
        
        # Process at different scales
        scale_features = []
        for scale_conv in self.scale_convs:
            scale_feat = scale_conv(x)
            scale_features.append(scale_feat)
        
        # Concatenate scale features
        concat_scales = torch.cat(scale_features, dim=1)  # [B, C*num_scales, H, W]
        
        # Compute scale attention (Eq. 6)
        scale_weights = self.scale_attention(concat_scales)  # [B, num_scales, 1, 1]
        
        # Weighted combination
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weight = scale_weights[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        multi_scale_features = sum(weighted_features)
        
        # Uncertainty estimation (Eq. 7)
        log_var = self.uncertainty_head(concat_scales)  # [B, num_scales, H, W]
        scale_uncertainty = torch.exp(log_var)
        
        return multi_scale_features, scale_uncertainty


class SharedDecoder(nn.Module):
    """
    Shared Decoder pathway for spatial resolution recovery.
    Implements efficient upsampling with skip connections.
    """
    def __init__(self, 
                 encoder_channels: List[int],
                 decoder_channels: List[int] = [256, 128, 64, 32],
                 num_classes: int = 6):
        super(SharedDecoder, self).__init__()
        
        # Decoder blocks with upsampling
        self.decoder_blocks = nn.ModuleList()
        
        in_channels = encoder_channels[-1]
        for out_channels in decoder_channels:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            )
            in_channels = out_channels
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused features [B, C, H, W]
        Returns:
            output: [B, num_classes, H_out, W_out]
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        output = self.classifier(x)
        
        return output


class FWDNNet(nn.Module):
    """
    FWDNNet: Fused Weights Deep Neural Tokenization Networks
    """
    def __init__(self,
                 num_classes: int = 6,
                 input_channels: int = 3,
                 pretrained: bool = True):
        super(FWDNNet, self).__init__()
        
        # Initialize heterogeneous encoders
        self.encoders = self._build_encoders(pretrained)
        
        # Get encoder output channels
        self.encoder_channels = [512, 768, 512, 384, 768]  # ResNet34, IncV3, VGG16, EfficientB3, SwinT
        
        # TensorDot Fusion Module (Eq. 2-3)
        self.tensordot_fusion = TensorDotFusion(
            input_channels=self.encoder_channels,
            output_channels=256
        )
        
        # Probabilistic Attention Weighting (Eq. 4-5)
        self.attention_weighting = ProbabilisticAttentionWeighting(
            input_channels=self.encoder_channels,
            hidden_dim=128
        )
        
        # Multi-Scale Feature Pyramid (Eq. 6-7)
        self.feature_pyramid = MultiScaleFeaturePyramid(
            channels=256,
            num_scales=4
        )
        
        # Shared Decoder
        self.decoder = SharedDecoder(
            encoder_channels=[256],
            decoder_channels=[256, 128, 64, 32],
            num_classes=num_classes
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_encoders(self, pretrained: bool) -> nn.ModuleDict:
        """Build heterogeneous encoder ensemble"""
        encoders = nn.ModuleDict()
        
        # 1. ResNet-34 Encoder
        resnet = models.resnet34(pretrained=pretrained)
        encoders['resnet34'] = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. Inception-V3 Encoder
        inception = models.inception_v3(pretrained=pretrained, aux_logits=False)
        encoders['inceptionv3'] = nn.Sequential(*list(inception.children())[:-1])
        
        # 3. VGG-16 Encoder
        vgg = models.vgg16(pretrained=pretrained)
        encoders['vgg16'] = vgg.features
        
        # 4. EfficientNet-B3 Encoder
        efficientnet = models.efficientnet_b3(pretrained=pretrained)
        encoders['efficientnet_b3'] = efficientnet.features
        
        # 5. Swin Transformer (placeholder - would need timm library)
        # For now, using a CNN approximation
        encoders['swin_t'] = self._build_swin_alternative()
        
        return encoders
    
    def _build_swin_alternative(self) -> nn.Module:
        """
        Simplified Swin Transformer alternative.
        In practice, use: timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        """
        return nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 192, 2, stride=2),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 384, 2, stride=2),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 768, 2, stride=2),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )
    
    def _initialize_weights(self):
        """Initialize weights for custom modules"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FWDNNet
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary containing:
                - output: Final segmentation map [B, num_classes, H, W]
                - attention_weights: Encoder attention weights [B, num_encoders]
                - uncertainty: Prediction uncertainty [B, num_scales, H, W]
        """
        B, C, H, W = x.shape
        
        # Extract features from all encoders
        encoder_features = []
        
        for name, encoder in self.encoders.items():
            if 'inception' in name:
                # InceptionV3 requires specific input size
                x_resized = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                feat = encoder(x_resized)
                feat = F.adaptive_avg_pool2d(feat, (H//32, W//32))
            else:
                feat = encoder(x)
            
            encoder_features.append(feat)
        
        # Compute probabilistic attention weights (Eq. 4-5)
        attention_weights, uncertainty_encoder = self.attention_weighting(
            encoder_features, 
            training=self.training
        )
        
        # Apply attention-weighted features
        weighted_features = []
        for i, feat in enumerate(encoder_features):
            weight = attention_weights[:, i:i+1, :, :]
            weighted_features.append(feat * weight)
        
        # TensorDot Fusion (Eq. 2-3)
        fused_features = self.tensordot_fusion(weighted_features)
        
        # Multi-Scale Feature Pyramid (Eq. 6-7)
        pyramid_features, scale_uncertainty = self.feature_pyramid(fused_features)
        
        # Shared Decoder
        output = self.decoder(pyramid_features)
        
        # Resize to input resolution
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return {
            'output': output,
            'attention_weights': attention_weights.squeeze(-1).squeeze(-1),
            'encoder_uncertainty': uncertainty_encoder,
            'scale_uncertainty': scale_uncertainty,
            'fused_features': fused_features
        }


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




