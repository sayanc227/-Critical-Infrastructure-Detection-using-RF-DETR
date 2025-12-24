import argparse
from rfdetr import RFDETRBase


def main(args):
model = RFDETRBase()


model.train(
dataset_dir=args.dataset_dir,
epochs=args.epochs,
batch_size=args.batch_size,
grad_accum_steps=args.grad_accum_steps,
lr=args.lr,
output_dir=args.output_dir,
resume=args.resume,
early_stopping=args.early_stopping
)


if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Train RF-DETR on COCO dataset")
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum_steps", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--early_stopping", action="store_true")


main(parser.parse_args())
