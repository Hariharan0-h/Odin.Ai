import torch
from model import Odin

def test_model():
    print("Testing model creation...")
    device = torch.device('cpu')
    
    model = Odin(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=2,
        d_ff=512,
        max_seq_len=64
    ).to(device)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("Testing forward pass...")
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print("Forward pass successful!")
    
    return True

if __name__ == "__main__":
    test_model()