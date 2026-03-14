import torch
import clip
from pathlib import Path
from PIL import Image
from backend.load_finetuned_clip import load_finetuned_model, test_model
from backend.config import MODEL_CHECKPOINT

def test_comprehensive():
    """Test the model comprehensively with only categories from training"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model
    if MODEL_CHECKPOINT is None:
        print("MODEL_CHECKPOINT is not set in config.py")
        return
    checkpoint_path = Path(MODEL_CHECKPOINT)
    model, preprocess = load_finetuned_model(checkpoint_path, device)

    # Also load base model for comparison
    print("\nLoading base (non-fine-tuned) CLIP model for comparison...")
    base_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    base_model.eval()

    # Test images from different categories
    test_cases = [
        ("sneakers-dataset/nike_air_force_1_low/0001.jpg", "Air Force 1 Low"),
        ("sneakers-dataset/nike_air_jordan_1_high/0001.jpg", "Air Jordan 1 High"),
        ("sneakers-dataset/nike_dunk_low/0001.jpg", "Dunk Low"),
    ]

    # Text queries - only categories that exist in training data
    text_queries = [
        "a photo of Air Force 1 Low sneakers",
        "a photo of Air Jordan 1 High sneakers",
        "a photo of Dunk Low sneakers",
        "a photo of Air Max 90 sneakers",
        "a photo of Cortez sneakers",
    ]

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*80)

    for img_path, expected_class in test_cases:
        test_image = Path(img_path)
        if not test_image.exists():
            print(f"\nSkipping {img_path} - file not found")
            continue

        print(f"\n{'='*80}")
        print(f"Testing: {img_path}")
        print(f"Expected: {expected_class}")
        print(f"{'='*80}")

        # Test with fine-tuned model
        print("\n--- FINE-TUNED MODEL ---")
        finetuned_probs = test_model(model, preprocess, test_image, text_queries, device)
        for query, prob in zip(text_queries, finetuned_probs):
            marker = "✓✓✓" if expected_class in query else "   "
            print(f"{marker} {query}: {prob*100:.2f}%")

        # Test with base model
        print("\n--- BASE MODEL (not fine-tuned) ---")
        base_probs = test_model(base_model, preprocess, test_image, text_queries, device)
        for query, prob in zip(text_queries, base_probs):
            marker = "✓✓✓" if expected_class in query else "   "
            print(f"{marker} {query}: {prob*100:.2f}%")

        # Compare improvement
        print("\n--- COMPARISON (Fine-tuned - Base) ---")
        expected_idx = next(i for i, q in enumerate(text_queries) if expected_class in q)
        improvement = (finetuned_probs[expected_idx] - base_probs[expected_idx]) * 100
        print(f"Improvement for correct class: {improvement:+.2f}%")
        if improvement < 1:
            print("⚠️  WARNING: Fine-tuning made things worse or had no effect!")

if __name__ == "__main__":
    test_comprehensive()
