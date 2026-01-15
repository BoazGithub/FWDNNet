# inference.py - Inference and evaluation script

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List
import cv2

class FWDNNetInference:
    """
    Inference pipeline for FWDNNet with visualization
    """
    def __init__(self,
                 model: nn.Module,
                 checkpoint_path: str,
                 device: torch.device,
                 num_classes: int = 6):
        
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class colors for visualization
        self.class_colors = self.get_class_colors()
        
    def get_class_colors(self) -> np.ndarray:
        """Define colors for each class"""
        colors = np.array([
            [128, 0, 0],      # Building - Dark Red
            [0, 128, 0],      # Land - Green
            [128, 128, 0],    # Road - Olive
            [0, 255, 0],      # Vegetation - Bright Green
            [0, 0, 255],      # Water - Blue
            [128, 128, 128]   # Other - Gray
        ])
        return colors
    
    @torch.no_grad()
    def predict(self, 
                image: np.ndarray,
                return_attention: bool = True) -> Dict:
        """
        Perform inference on a single image: for check man before
        
        Args:
            image: Input RGB image [H, W, 3]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and visualizations
        """
        # Preprocess
        original_size = image.shape[:2]
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor)
        
        # Get predictions
        pred_logits = outputs['output']
        pred_masks = torch.argmax(pred_logits, dim=1)[0]  # [H, W]
        pred_probs = F.softmax(pred_logits, dim=1)[0]      # [C, H, W]
        
        # Resize to original size
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(0).unsqueeze(0).float(),
            size=original_size,
            mode='nearest'
        )[0, 0].long()
        
        pred_probs = F.interpolate(
            pred_probs.unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        )[0]
        
        results = {
            'pred_mask': pred_masks.cpu().numpy(),
            'pred_probs': pred_probs.cpu().numpy(),
            'colored_mask': self.colorize_mask(pred_masks.cpu().numpy()),
        }
        
        if return_attention:
            results['attention_weights'] = outputs['attention_weights'][0].cpu().numpy()
            results['encoder_uncertainty'] = outputs['encoder_uncertainty'][0].cpu().numpy()
        
        return results
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # To tensor [C, H, W]
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        return image
    
    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert class indices to colored visualization"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for cls_id in range(self.num_classes):
            colored[mask == cls_id] = self.class_colors[cls_id]
        
        return colored
    
    def visualize_results(self,
                         image: np.ndarray,
                         results: Dict,
                         save_path: str = None):
        """
        Create comprehensive visualization
        
        Creates figure with:
        - Original image
        - Predicted segmentation
        - Confidence map
        - Attention weights visualization
        """
        fig = plt.figure(figsize=(20, 10))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Predicted segmentation
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(results['colored_mask'])
        ax2.set_title('Predicted Segmentation', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = plt.subplot(2, 3, 3)
        overlay = cv2.addWeighted(image, 0.6, results['colored_mask'], 0.4, 0)
        ax3.imshow(overlay)
        ax3.set_title('Overlay', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Confidence map
        ax4 = plt.subplot(2, 3, 4)
        confidence = results['pred_probs'].max(axis=0)
        im = ax4.imshow(confidence, cmap='jet', vmin=0, vmax=1)
        ax4.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # Attention weights
        if 'attention_weights' in results:
            ax5 = plt.subplot(2, 3, 5)
            encoder_names = ['ResNet34', 'InceptionV3', 'VGG16', 'EfficientB3', 'Swin-T']
            attention = results['attention_weights']
            
            bars = ax5.bar(encoder_names, attention)
            ax5.set_ylim([0, 1])
            ax5.set_title('Encoder Attention Weights', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Weight', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Color bars
            colors = plt.cm.viridis(attention / attention.max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Class distribution
        ax6 = plt.subplot(2, 3, 6)
        class_names = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Other']
        pred_mask = results['pred_mask']
        
        class_counts = []
        for cls_id in range(self.num_classes):
            count = (pred_mask == cls_id).sum() / pred_mask.size * 100
            class_counts.append(count)
        
        bars = ax6.barh(class_names, class_counts)
        ax6.set_xlabel('Percentage (%)', fontsize=12)
        ax6.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Color bars by class
        for bar, color in zip(bars, self.class_colors):
            bar.set_color(color / 255.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_inference(self,
                       image_paths: List[str],
                       output_dir: str):
        """
        Perform batch inference on multiple images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for img_path in tqdm(image_paths, desc='Processing images'):
            # Load image
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # Predict
            results = self.predict(image)
            
            # Save results
            basename = os.path.basename(img_path).split('.')[0]
            
            # Save colored mask
            mask_path = os.path.join(output_dir, f'{basename}_mask.png')
            Image.fromarray(results['colored_mask']).save(mask_path)
            
            # Save visualization
            viz_path = os.path.join(output_dir, f'{basename}_viz.png')
            self.visualize_results(image, results, save_path=viz_path)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(pred: np.ndarray, 
                   target: np.ndarray,
                   num_classes: int) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics | i personally advice to chose neceesary metrics, for me because it was winter time
    
    Returns metrics from Table III of the paper:
    - Overall Accuracy
    - Mean IoU
    - Class-wise IoU
    - F1-Score
    - Precision
    - Recall
    - Kappa Coefficient
    """
    metrics = {}
    
    # Overall Accuracy (Eq. 15)
    correct = (pred == target).sum()
    total = pred.size
    metrics['overall_accuracy'] = correct / total
    
    # Per-class metrics
    class_ious = []
    class_f1s = []
    class_precisions = []
    class_recalls = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # True Positives, False Positives, False Negatives
        tp = np.logical_and(pred_cls, target_cls).sum()
        fp = np.logical_and(pred_cls, ~target_cls).sum()
        fn = np.logical_and(~pred_cls, target_cls).sum()
        
        # IoU (Eq. 16)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        class_ious.append(iou)
        
        # Precision and Recall (Eq. 19-20)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_precisions.append(precision)
        class_recalls.append(recall)
        
        # F1-Score (Eq. 18)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_f1s.append(f1)
    
    # Mean IoU (Eq. 16)
    metrics['mean_iou'] = np.mean(class_ious)
    metrics['class_ious'] = class_ious
    
    # Average F1, Precision, Recall
    metrics['f1_score'] = np.mean(class_f1s)
    metrics['precision'] = np.mean(class_precisions)
    metrics['recall'] = np.mean(class_recalls)
    
    # Kappa Coefficient (Eq. 21)
    po = metrics['overall_accuracy']
    pe = sum([(pred == cls).sum() * (target == cls).sum() 
              for cls in range(num_classes)]) / (total ** 2)
    metrics['kappa'] = (po - pe) / (1 - pe) if pe < 1 else 0
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(num_classes=6, pretrained=False)
    
    # Initialize inference
    inferencer = FWDNNetInference(
        model=model,
        checkpoint_path='./checkpoints/best_model.pth  #for your implimentation please chose basedon your directory',
        device=device,
        num_classes=6
    )
    
    # Single image inference
    image_path = 'path/to/test/image.jpg'
    image = np.array(Image.open(image_path).convert('RGB'))
    
    results = inferencer.predict(image, return_attention=True)
    inferencer.visualize_results(image, results, save_path='result.png')
    
    # Print attention weights
    print("\nEncoder Attention Weights:")
    encoder_names = ['ResNet34', 'InceptionV3', 'VGG16', 'EfficientB3', 'Swin-T']
    for name, weight in zip(encoder_names, results['attention_weights']):
        print(f"  {name}: {weight:.3f}")