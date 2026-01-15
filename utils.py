# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_model(num_classes: int = 6, pretrained: bool = True) -> FWDNNet:
    """Factory function to create FWDNNet model"""
    model = FWDNNet(num_classes=num_classes, input_channels=3, pretrained=pretrained)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create model
    model = create_model(num_classes=6, pretrained=False)
    
    print(f"Total parameters: {count_parameters(model) / 1e6:.2f}M")
    
    # Example input
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    target = torch.randint(0, 6, (batch_size, 512, 512))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Loss computation
    criterion = ComprehensiveLoss(num_classes=6)
    
    model.train()
    outputs = model(x)
    
    # Get encoder features for diversity loss (you'd need to modify forward pass to return these)
    losses = criterion(
        pred=outputs['output'],
        target=target,
        attention_weights=outputs['attention_weights'],
        encoder_features=None,  # Would need to extract from model
        uncertainty=outputs['scale_uncertainty']
    )
    
    print("\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")