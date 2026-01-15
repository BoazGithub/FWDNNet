# train.py - Complete training pipeline for FWDNNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RemoteSensingDataset(Dataset):
    """
    Dataset class for remote sensing imagery semantic segmentation
    """
    def __init__(self, 
                 image_paths: list,
                 mask_paths: list,
                 transform=None,
                 num_classes: int = 6):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask (implement your loading logic)
        # image = load_image(self.image_paths[idx])
        # mask = load_mask(self.mask_paths[idx])
        
        # Placeholder
        image = np.random.rand(512, 512, 3).astype(np.float32)
        mask = np.random.randint(0, self.num_classes, (512, 512)).astype(np.int64)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def get_transforms(training: bool = True):
    """
    Data augmentation strategy from paper Section 3.2
    """
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GaussNoise(var_limit=(0.0, 0.02), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class FWDNNetTrainer:
    """
    Complete training pipeline for FWDNNet
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 device: torch.device,
                 config: Dict):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Optimizer with AdamW (Table II from paper)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler (exponential decay)
        self.scheduler = ExponentialLR(
            self.optimizer,
            gamma=0.95  # From Table II
        )
        
        # Metrics tracking
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'segmentation': 0.0,
            'consistency': 0.0,
            'uncertainty': 0.0,
            'diversity': 0.0,
            'sparsity': 0.0,
            'boundary': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute losses
            losses = self.criterion(
                pred=outputs['output'],
                target=masks,
                attention_weights=outputs['attention_weights'],
                encoder_features=None,
                uncertainty=outputs['scale_uncertainty']
            )
            
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        num_samples = 0
        
        for images, masks in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(
                pred=outputs['output'],
                target=masks,
                attention_weights=outputs['attention_weights'],
                encoder_features=None,
                uncertainty=outputs['scale_uncertainty']
            )
            
            total_loss += losses['total'].item()
            
            # Compute mIoU
            pred_masks = torch.argmax(outputs['output'], dim=1)
            iou = self.compute_miou(pred_masks, masks)
            total_iou += iou * images.size(0)
            num_samples += images.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_miou = total_iou / num_samples
        
        return avg_loss, avg_miou
    
    def compute_miou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute mean Intersection over Union"""
        num_classes = self.config['num_classes']
        ious = []
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = np.logical_and(pred_cls, target_cls).sum()
            union = np.logical_or(pred_cls, target_cls).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def train(self):
        """Complete training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model) / 1e6:.2f}M")
        
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validate
            val_loss, val_miou = self.validate()
            self.val_losses.append(val_loss)
            self.val_mious.append(val_miou)
            
            # Learning rate scheduling (step every 10 epochs as per Table II)
            if (epoch + 1) % 10 == 0:
                self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, 'best_model.pth')
                patience_counter = 0
                print(f"✓ New best model saved! mIoU: {val_miou:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping (patience=20 from Table II)
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\nTraining completed!")
        print(f"Best validation mIoU: {self.best_miou:.4f}")
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Configuration (matching Table II from paper)
    config = {
        'num_classes': 6,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 1e-3,
        'early_stopping_patience': 20,
        'checkpoint_dir': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create datasets
    # Replace with your actual data paths
    train_images = []  # List of training image paths
    train_masks = []   # List of training mask paths
    val_images = []    # List of validation image paths
    val_masks = []     # List of validation mask paths
    
    train_dataset = RemoteSensingDataset(
        train_images,
        train_masks,
        transform=get_transforms(training=True),
        num_classes=config['num_classes']
    )
    
    val_dataset = RemoteSensingDataset(
        val_images,
        val_masks,
        transform=get_transforms(training=False),
        num_classes=config['num_classes']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=config['num_classes'], pretrained=True)
    
    # Create loss criterion
    criterion = ComprehensiveLoss(
        num_classes=config['num_classes'],
        lambda_consistency=0.1,
        lambda_uncertainty=0.05,
        lambda_diversity=0.1,
        lambda_sparsity=0.01,
        lambda_boundary=0.2
    )
    
    # Create trainer
    device = torch.device(config['device'])
    trainer = FWDNNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=config
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()