import os
import torch
import cv2
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys

# Add parent directory to path BEFORE any local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model building logic from the source
# Use relative imports since we're inside src/
from models import build_model 
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from datasets import build_dataset

def main(args):
    # 1. Load Configuration from Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # 2. Build Model
    # Create a config-like object for build_model
    # RF-DETR's build_model typically expects an args object with specific fields
    try:
        model, criterion, postprocessors = build_model(args)
    except Exception as e:
        print(f"Error building model: {e}")
        print("Trying alternative model building approach...")
        # Alternative: Load config from checkpoint if available
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key, value in vars(saved_args).items():
                if not hasattr(args, key):
                    setattr(args, key, value)
        model, criterion, postprocessors = build_model(args)
    
    model.to(args.device)
    
    # 3. Load Weights
    print("Loading model weights...")
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully!")

    # 4. Prepare Transformation
    transform = T.Compose([
        T.Resize((args.resolution, args.resolution)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Process Images
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(source_path.glob(ext)))
    
    if len(image_paths) == 0:
        print(f"No images found in {source_path}")
        return
    
    print(f"Found {len(image_paths)} images. Starting inference...")

    with torch.no_grad():
        for idx, img_p in enumerate(image_paths):
            print(f"Processing [{idx+1}/{len(image_paths)}]: {img_p.name}")
            
            try:
                # Load image
                img_raw = Image.open(img_p).convert("RGB")
                w, h = img_raw.size
                img = transform(img_raw).unsqueeze(0).to(args.device)
                
                # Inference
                outputs = model(img)
                
                # Process results (BBoxes & Scores)
                orig_target_sizes = torch.tensor([[h, w]]).to(args.device)
                results = postprocessors['bbox'](outputs, orig_target_sizes)[0]
                
                # Filter by confidence
                scores = results['scores']
                labels = results['labels']
                boxes = results['boxes']
                
                keep = scores > args.conf
                filt_scores = scores[keep]
                filt_labels = labels[keep]
                filt_boxes = boxes[keep]
                
                print(f"  Detected {len(filt_scores)} objects above threshold {args.conf}")
                
                # Visualizing
                vsl = COCOVisualizer()
                
                # Prepare detection dict
                detections = {
                    'boxes': filt_boxes.cpu().numpy(),
                    'labels': filt_labels.cpu().numpy(),
                    'scores': filt_scores.cpu().numpy()
                }
                
                # Convert PIL to numpy for visualization
                import numpy as np
                img_np = np.array(img_raw)
                
                # Draw boxes on image
                output_img = vsl.visualize(img_np, detections, args.class_names)
                
                # Save
                save_name = output_path / img_p.name
                if isinstance(output_img, np.ndarray):
                    output_img = Image.fromarray(output_img)
                output_img.save(save_name)
                print(f"  Saved: {save_name}")
                
            except Exception as e:
                print(f"  Error processing {img_p.name}: {e}")
                continue

    print(f"\nInference complete! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('RF-DETR inference')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to checkpoint file')
    parser.add_argument('--source', required=True, type=str, help='Path to input images')
    parser.add_argument('--output', default='results', type=str, help='Output directory')
    parser.add_argument('--conf', default=0.35, type=float, help='Confidence threshold')
    parser.add_argument('--resolution', default=560, type=int, help='Input resolution')
    
    # RF-DETR model architecture args
    parser.add_argument('--modelname', default='rfdetr', type=str)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--encoder', default='dinov2_windowed_small', type=str)
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--with_box_refine', default=True, type=bool)
    parser.add_argument('--two_stage', default=False, type=bool)
    parser.add_argument('--use_nms', default=True, type=bool)
    
    args = parser.parse_args()
    
    # Apply your specific class names
    args.class_names = [
        'Airport_Runway', 'Bridge', 'Cargo_Ship', 'Cooling_Tower', 'Dam', 
        'Electrical_Substation', 'Energy Storage Infrastructure', 'Mobile Tower', 
        'Nuclear_Reactor', 'Oil Refinery', 'Satellite_Dish /Ground_Station', 
        'Seaport', 'Shipping Containers', 'Solar_Power_Plant', 'Thermal Power Plant', 
        'Transmission Tower', 'Water Tower', 'Wind Turbine', 'mobile harbour cranes'
    ]
    
    main(args)
