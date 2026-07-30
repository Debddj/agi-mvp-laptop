import torch

from model.transformer import MultimodalLLM
from model.vision import VisionEncoder, process_image


def test_vision_encoder_output_shape():
    encoder = VisionEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        vision_hidden_dim=128,
        llm_hidden_dim=256,
        n_layers=2,
        n_heads=4
    )
    images = torch.randn(2, 3, 224, 224)
    out = encoder(images)
    expected_num_patches = (224 // 16) ** 2
    assert out.shape == (2, expected_num_patches, 256)


def test_process_image_tensor():
    img_tensor = torch.randn(3, 100, 100)
    processed = process_image(img_tensor, image_size=224)
    assert processed.shape == (1, 3, 224, 224)


def test_multimodal_llm_forward():
    config = {
        "vocab_size": 200,
        "n_layers": 2,
        "n_heads": 4,
        "hidden_size": 128,
        "mlp_ratio": 4,
        "max_seq_len": 512,
        "image_size": 224,
        "patch_size": 16,
        "vision_hidden_dim": 128,
        "vision_layers": 2,
        "vision_heads": 4,
    }
    model = MultimodalLLM(config)
    x = torch.randint(0, 200, (2, 10))
    images = torch.randn(2, 3, 224, 224)

    # Multi-modal prefill pass
    logits = model(x, images=images)
    num_patches = (224 // 16) ** 2
    expected_total_seq = num_patches + 10
    assert logits.shape == (2, expected_total_seq, 200)
