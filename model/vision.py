import torch
import torch.nn as nn
import torch.nn.functional as F
from model.norms import RMSNorm


class PatchEmbedding(nn.Module):
    """
    Converts 2D image tensors [B, C, H, W] into 1D patch embeddings [B, N_patches, vision_hidden_dim].
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Input image height/width ({H}x{W}) must match expected ({self.image_size}x{self.image_size})"
        )
        # [B, embed_dim, grid, grid] -> [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x


class VisionEncoder(nn.Module):
    """
    Vision Encoder module that encodes raw images into visual tokens projected to LLM hidden dimension.
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        vision_hidden_dim=384,
        llm_hidden_dim=384,
        n_layers=4,
        n_heads=6
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_hidden_dim
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_hidden_dim,
            nhead=n_heads,
            dim_feedforward=vision_hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = RMSNorm(vision_hidden_dim)
        
        # Projection head to map visual features to LLM hidden dimension
        if vision_hidden_dim != llm_hidden_dim:
            self.proj = nn.Linear(vision_hidden_dim, llm_hidden_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, images):
        # images: [B, C, H, W]
        x = self.patch_embed(images)  # [B, num_patches, vision_hidden_dim]
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x)  # [B, num_patches, llm_hidden_dim]
        return x


def process_image(image_input, image_size=224):
    """
    Helper function to preprocess image inputs (PIL Image, numpy array, or torch.Tensor)
    into standard normalized PyTorch FloatTensor [B, 3, image_size, image_size].
    """
    if isinstance(image_input, torch.Tensor):
        if image_input.ndim == 3:
            image_input = image_input.unsqueeze(0)
        if image_input.shape[-2:] != (image_size, image_size):
            image_input = F.interpolate(
                image_input, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
        return image_input.float()

    # Try PIL Image or numpy array
    try:
        from PIL import Image
        import numpy as np
        
        if isinstance(image_input, str):
            image_input = Image.open(image_input).convert("RGB")
        elif hasattr(image_input, "convert"):
            image_input = image_input.convert("RGB")
            
        if hasattr(image_input, "resize"):
            image_input = image_input.resize((image_size, image_size))
            
        arr = np.array(image_input, dtype=np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr.transpose(2, 0, 1)  # HWC to CHW
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 3, H, W]
        return tensor
    except Exception as e:
        # Fallback dummy zero image
        return torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
