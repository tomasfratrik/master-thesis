import argparse
import json
from pathlib import Path

from finetuned_classifier_service import FineTunedSneakerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sneaker label top-k scores from a fine-tuned checkpoint."
    )
    parser.add_argument(
        "--image",
        required=True,
        nargs="+",
        help="One or more input image paths representing the same sneaker item.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = [Path(image_path) for image_path in args.image]
    checkpoint_path = Path(args.checkpoint)

    missing_paths = [str(image_path) for image_path in image_paths if not image_path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Image not found: {', '.join(missing_paths)}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)
    result = classifier.predict_image_paths(image_paths=image_paths, k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
