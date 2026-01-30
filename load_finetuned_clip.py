import torch
import clip
from pathlib import Path
from PIL import Image

def load_finetuned_model(checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a fine-tuned CLIP model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on

    Returns:
        model: Fine-tuned CLIP model
        preprocess: CLIP preprocessing function
    """
    # Load base model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Loaded fine-tuned model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return model, preprocess

def test_model(model, preprocess, image_path, text_queries, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Test the model on an image with multiple text queries

    Args:
        model: CLIP model
        preprocess: Preprocessing function
        image_path: Path to test image
        text_queries: List of text queries to compare
        device: Device

    Returns:
        probabilities: Softmax probabilities for each query
    """
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize text
    text = clip.tokenize(text_queries).to(device)

    # Get predictions
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity[0].cpu().numpy()

if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model
    checkpoint_path = Path("artifacts/finetuned_models/clip_nike_best.pt")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python finetune_clip_nike.py")
        exit(1)

    model, preprocess = load_finetuned_model(checkpoint_path, device)

    # Test on a sample Nike image
    test_image = Path("sneakers-dataset/nike_air_force_1_low/0001.jpg")

    if test_image.exists():
        # IMPORTANT: These text queries must match the format used during training
        # Training uses: nike_dir.name.replace('_', ' ').replace('nike ', '').title()
        # This removes "nike " prefix and title-cases the rest
        text_queries = [
            "a photo of Air Force 1 Low sneakers",
            "a photo of Air Jordan 1 High sneakers",
            "a photo of Ultraboost sneakers",
            "a photo of Chuck Taylor sneakers"
        ]

        probabilities = test_model(model, preprocess, test_image, text_queries, device)

        print(f"\nTest image: {test_image}")
        print("\nPredictions:")
        for query, prob in zip(text_queries, probabilities):
            print(f"{query}: {prob*100:.2f}%")
