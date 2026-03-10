import argparse
import json
from pathlib import Path

from finetuned_classifier_service import FineTunedSneakerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sneaker label top-k scores from a fine-tuned checkpoint."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)
    result = classifier.predict_image_path(image_path=image_path, k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
