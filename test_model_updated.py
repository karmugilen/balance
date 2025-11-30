#!/usr/bin/env python3
"""
Test script to load and use the updated ILWT model with learnable parameters.
This demonstrates how to properly load the model from checkpoints and use it for
embedding secrets and extracting them.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add parent directory to path to import from ilwt_main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the model classes from ilwt_main
from ilwt_main import (
    StarINNWithILWT, 
    LearnableSubbandWeights,
    LearnableYCbCrScaling
)


def load_image(path, size=224):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    return tensor


def denormalize_to_pil(tensor):
    """Convert tensor back to PIL image"""
    tensor = (tensor / 2.0) + 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = transforms.ToPILImage()(tensor.cpu())
    return img


def load_model(model_path, device='cuda'):
    """
    Load model from checkpoint or state_dict.
    Handles both old format (just state_dict) and new format (full checkpoint).
    """
    # Create model with default architecture
    model = StarINNWithILWT(
        channels=6,
        num_blocks=8,  # Match training config
        hidden_channels=128,  # Match training config
        transform_type="ilwt53"
    )
    
    # Load checkpoint (weights_only=False needed for numpy objects)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New checkpoint format
        print("Loading from full checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Loaded model from epoch {checkpoint['epoch']}")
        if 'recovery_psnr' in checkpoint:
            print(f"Recovery PSNR: {checkpoint['recovery_psnr']:.2f} dB")
    else:
        # Old format (just state_dict)
        print("Loading from state_dict")
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def embed_secret(model, cover_path, secret_path, output_path, device='cuda'):
    """
    Embed secret image into cover image.
    """
    # Load images
    cover_tensor = load_image(cover_path).unsqueeze(0).to(device)
    secret_tensor = load_image(secret_path).unsqueeze(0).to(device)
    
    # Concatenate cover and secret
    input_tensor = torch.cat([cover_tensor, secret_tensor], dim=1)
    
    # Forward pass = embedding
    with torch.no_grad():
        stego_output, _ = model(input_tensor)
        stego_host = stego_output[:, :3, :, :][0]  # Extract cover channels
    
    # Save stego image
    stego_img = denormalize_to_pil(stego_host)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    stego_img.save(output_path)
    print(f"✅ Saved stego image to {output_path}")
    
    return stego_img


def extract_secret(model, stego_path, output_path, device='cuda'):
    """
    Extract secret image from stego image.
    """
    # Load stego image
    stego_tensor = load_image(stego_path).unsqueeze(0).to(device)
    
    # Create stego-like input (stego + zeros for secret channels)
    stego_like = torch.cat([stego_tensor, torch.zeros_like(stego_tensor)], dim=1)
    
    # Inverse pass = extraction
    with torch.no_grad():
        reconstructed_input = model.inverse(stego_like)
        recovered_secret = reconstructed_input[:, 3:, :, :][0]  # Extract secret channels
    
    # Save recovered secret
    recovered_img = denormalize_to_pil(recovered_secret)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    recovered_img.save(output_path)
    print(f"✅ Saved recovered secret to {output_path}")
    
    return recovered_img


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ILWT model with learnable parameters")
    parser.add_argument("--model", default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["embed", "extract", "both"], default="both", help="Operation mode")
    parser.add_argument("--cover", help="Path to cover image (for embed mode)")
    parser.add_argument("--secret", help="Path to secret image (for embed mode)")
    parser.add_argument("--stego", help="Path to stego image (for extract mode)")
    parser.add_argument("--output", default="test_output", help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, device)
    print("✅ Model loaded successfully")
    
    # Print learnable parameters
    with torch.no_grad():
        kY, kC = model.ycbcr_scaling.get_scales()
        weights, wLL2 = model.subband_weights.get_weights()
        print(f"\n📊 Learnable Parameters:")
        print(f"  YCbCr Scaling: kY={kY.item():.4f}, kC={kC.item():.4f}")
        print(f"  Wavelet Weights: LL={weights[0].item():.4f}, LH={weights[1].item():.4f}, " 
              f"HL={weights[2].item():.4f}, HH={weights[3].item():.4f}, LL2={wLL2.item():.4f}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Run operations
    if args.mode in ["embed", "both"]:
        if not args.cover or not args.secret:
            print("\n⚠️  Need --cover and --secret for embedding")
            # Use first available test images if not provided
            test_cover = "my_images/test_cover.jpg" if os.path.exists("my_images/test_cover.jpg") else None
            test_secret = "my_images/test_secret.jpg" if os.path.exists("my_images/test_secret.jpg") else None
            
            if test_cover and test_secret:
                print(f"Using test images: {test_cover}, {test_secret}")
                args.cover = test_cover
                args.secret = test_secret
                args.stego = os.path.join(args.output, "stego.png")
            else:
                print("No test images found. Skipping embed.")
                return
        
        print(f"\n🔒 Embedding secret...")
        stego_path = args.stego or os.path.join(args.output, "stego.png")
        embed_secret(model, args.cover, args.secret, stego_path, device)
        args.stego = stego_path
    
    if args.mode in ["extract", "both"]:
        if not args.stego:
            print("\n⚠️  Need --stego for extraction")
            return
        
        print(f"\n🔓 Extracting secret...")
        extract_secret(model, args.stego, os.path.join(args.output, "recovered_secret.png"), device)
    
    print("\n✅ All operations completed successfully!")


if __name__ == "__main__":
    main()
