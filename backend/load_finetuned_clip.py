import torch
from pathlib import Path
from PIL import Image
from backend.config import MODEL_CHECKPOINT, MODEL_USE_CHECKPOINT
from backend.model_loader import load_encoder

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
    # Load base model (no fine-tuned weights yet)
    encoder = load_encoder(
        device=torch.device(device),
        use_checkpoint=False,
        checkpoint_path=checkpoint_path,
        checkpoint_map_location=device,
        checkpoint_strict=False,
    )
    model = encoder.model
    preprocess = encoder.preprocess

    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    encoder = load_encoder(device=torch.device(device), use_checkpoint=False)
    encoder.model = model
    text = encoder.tokenize_texts(text_queries)

    # Get predictions
    with torch.no_grad():
        image_features = encoder.encode_image_tensors(image)
        text_features = encoder.encode_text_tokens(text)

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
    if MODEL_CHECKPOINT is None:
        print("MODEL_CHECKPOINT is not set in config.py")
        exit(1)
    checkpoint_path = Path(MODEL_CHECKPOINT)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python -m backend.finetune_clip")
        exit(1)

    if not MODEL_USE_CHECKPOINT:
        print("NOTE: MODEL_USE_CHECKPOINT is False. Set it to True in config.py to use checkpoint weights.")
    model, preprocess = load_finetuned_model(checkpoint_path, device)

    # Test on a sample sneaker image
    test_image = Path("sneakers-dataset/nike_air_force_1_low/0001.jpg")

    if test_image.exists():
        # IMPORTANT: These text queries must match the format used during training
        # Training uses: class_dir.name.replace('_', ' ').title()
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
