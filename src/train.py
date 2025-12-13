import argparse
import torch
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import RFDETRBase, handling potential installation issues
try:
    from rfdetr import RFDETRBase
except ImportError:
    print("❌ Error: 'rfdetr' library not found. Please ensure it is installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RF-DETR on custom dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory (COCO format)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("--- Initializing Training Pipeline ---")
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: GPU not found. Training will be extremely slow.")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing RFDETRBase model...")
    model = RFDETRBase()
    
    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"   Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum_steps}")
    
    try:
        model.train(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            output_dir=args.output_dir,
            save_interval=5  # Saves a checkpoint every 5 epochs
        )
        print("✅ Training Complete.")
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()
