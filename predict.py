import argparse
import json
from pathlib import Path

from sneaker_service import SneakerLabelService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sneaker label from an image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k predictions to return.")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Force loading checkpoint weights regardless of config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    use_checkpoint = True if args.use_checkpoint else None
    service = SneakerLabelService(use_checkpoint=use_checkpoint)
    result = service.predict_image_path(image_path=image_path, k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
