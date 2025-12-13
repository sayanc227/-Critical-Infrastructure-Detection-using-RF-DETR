import argparse
import os
import torch
import sys
import supervision as sv
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rfdetr import RFDETRBase
except ImportError:
    print("❌ Error: 'rfdetr' library not found.")
    sys.exit(1)

from models.helpers import patch_model_architecture, smart_load_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR mAP on validation set.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--conf', type=float, default=0.30, help='Confidence threshold for evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define Paths (Auto-detecting valid folder)
    # Standard COCO structure usually has a 'valid' folder
    valid_images_path = os.path.join(args.dataset_dir, "valid") 
    valid_ann_path = os.path.join(args.dataset_dir, "valid", "_annotations.coco.json")
    
    # Check if annotations exist in standard location, otherwise check root
    if not os.path.exists(valid_ann_path):
         root_ann = os.path.join(args.dataset_dir, "_annotations.coco.json")
         if os.path.exists(root_ann):
             print(f"⚠️ 'valid/_annotations.coco.json' not found. Using root annotation: {root_ann}")
             valid_ann_path = root_ann
         else:
             print(f"❌ Error: Could not find _annotations.coco.json in {valid_images_path} or root.")
             return

    print(f"Loading dataset from: {valid_ann_path}")
    
    # 2. Load Dataset using Supervision
    # This reads the ground truth labels
    try:
        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=valid_images_path,
            annotations_path=valid_ann_path
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 3. Initialize Model
    num_classes = len(dataset.classes)
    print(f"Classes found in dataset: {num_classes}")
    
    try:
        model = RFDETRBase(num_classes=num_classes)
    except TypeError:
        model = RFDETRBase()
    
    # Apply our custom fixes
    patch_model_architecture(model, num_classes)
    smart_load_weights(model, args.checkpoint, device)
    
    # 4. Run Evaluation Loop
    predictions = []
    print("Running inference on validation set...")
    
    # Iterate over dataset and predict
    # Dataset yields (path, image_numpy, annotations)
    for _, image, _ in tqdm(dataset):
        # Convert numpy image to PIL for model prediction
        from PIL import Image
        image_pil = Image.fromarray(image)
        
        with torch.no_grad():
            detections = model.predict(image_pil, threshold=args.conf)
            
        predictions.append(detections)

    # 5. Calculate mAP
    print("Calculating mAP...")
    try:
        mean_average_precision = sv.MeanAveragePrecision.from_detections(
            predictions=predictions,
            targets=dataset.annotations,
            classes=dataset.classes
        )

        print("\n================================================")
        print(f"mAP_50 (IOU=0.50): {mean_average_precision.map50:.4f}")
        print(f"mAP_75 (IOU=0.75): {mean_average_precision.map75:.4f}")
        print(f"mAP_50_95        : {mean_average_precision.map50_95:.4f}")
        print("================================================")
    except Exception as e:
         print(f"❌ Error calculating mAP: {e}")

if __name__ == "__main__":
    main()
