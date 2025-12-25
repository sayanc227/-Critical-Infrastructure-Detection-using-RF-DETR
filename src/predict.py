import argparse
import sys
from rfdetr import RFDETRBase
from PIL import Image
import os

def main(args):
    # instantiate model
    model = RFDETRBase()

    # Try to load checkpoint if provided.
    if args.checkpoint:
        # Some projects use resume=checkpoint with epochs=0 as a hack to load weights.
        # Try that first, then fall back to other common loaders if available.
        loaded = False
        try:
            model.train(dataset_dir=args.dataset_dir if args.dataset_dir else None,
                        epochs=0,
                        resume=args.checkpoint)
            loaded = True
        except Exception:
            # fallback attempts
            if hasattr(model, "load"):
                try:
                    model.load(args.checkpoint)
                    loaded = True
                except Exception:
                    loaded = False
            elif hasattr(model, "load_weights"):
                try:
                    model.load_weights(args.checkpoint)
                    loaded = True
                except Exception:
                    loaded = False

        if not loaded:
            print(f"Warning: could not load checkpoint '{args.checkpoint}'. Continuing without loading weights.", file=sys.stderr)

    # Open image
    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error: failed to open image '{args.image_path}': {e}", file=sys.stderr)
        sys.exit(2)

    # Run prediction (try with provided threshold, then without if that fails)
    try:
        detections = model.predict(image, threshold=args.threshold)
    except TypeError:
        # maybe the model.predict doesn't accept threshold kwarg
        try:
            detections = model.predict(image)
        except Exception as e:
            print(f"Error: prediction failed: {e}", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(f"Error: prediction failed: {e}", file=sys.stderr)
        sys.exit(3)

    # Print detections in a robust way
    print("Detections:")

    # Common expected structure: detections.class_id, detections.confidence, detections.xyxy
    try:
        if hasattr(detections, "class_id") and hasattr(detections, "confidence") and hasattr(detections, "xyxy"):
            cls_list = list(detections.class_id)
            conf_list = list(detections.confidence)
            boxes = list(detections.xyxy)
            # Validate lengths
            n = min(len(cls_list), len(conf_list), len(boxes))
            for i in range(n):
                cls = cls_list[i]
                conf = conf_list[i]
                box = boxes[i]
                # Convert box elements to floats if possible
                try:
                    box_vals = [float(x) for x in box]
                except Exception:
                    box_vals = list(box)
                # Format confidence safely
                try:
                    conf_val = float(conf)
                    print(f"Class {cls} | Conf {conf_val:.2f} | Box {box_vals}")
                except Exception:
                    print(f"Class {cls} | Conf {conf} | Box {box_vals}")
        else:
            # Fallback: if detections is iterable (list of detections), print each entry
            if isinstance(detections, (list, tuple)):
                for det in detections:
                    print(det)
            else:
                # Unknown structure; print repr
                print(repr(detections))
    except Exception as e:
        print(f"Error while formatting detections: {e}", file=sys.stderr)
        print("Raw detections:", repr(detections))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with RF-DETR")
    parser.add_argument("--dataset_dir", required=False, help="Path to dataset (only needed if training/resuming)")
    parser.add_argument("--checkpoint", required=False, help="Path to checkpoint to load (optional for inference)")
    parser.add_argument("--image_path", required=True, help="Path to image for inference")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)
