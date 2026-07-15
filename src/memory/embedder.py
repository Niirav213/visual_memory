"""
Visual Memory AI — CLIP Visual Embedder
Converts object crops and text queries into 512-dimensional embeddings
using OpenAI CLIP for semantic similarity matching.
"""

from typing import Union
import numpy as np
import torch
import clip
from PIL import Image


class VisualEmbedder:
    """
    CLIP-based visual and text embedding generator.

    Provides a shared embedding space where images and text can be
    compared using cosine similarity. This enables the system to
    match natural language queries against stored visual memories.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
    ):
        """
        Initialize the CLIP embedder.

        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
            device: Compute device ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Get embedding dimension from model
        self.embedding_dim = self.model.visual.output_dim

        print(f"[Embedder] Loaded CLIP {model_name} on {self.device} | Dim: {self.embedding_dim}")

    @torch.no_grad()
    def embed_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate a normalized embedding for an image or object crop.

        Args:
            image: BGR numpy array (from OpenCV) or PIL Image

        Returns:
            Normalized 512-dimensional embedding as numpy float32 array
        """
        # Convert numpy BGR to PIL RGB
        if isinstance(image, np.ndarray):
            if image.size == 0:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            # BGR → RGB conversion
            image_rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Preprocess and encode
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)

        # Normalize to unit vector for cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding for a text query.

        Args:
            text: Natural language query string

        Returns:
            Normalized 512-dimensional embedding as numpy float32 array
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        features = self.model.encode_text(text_tokens)

        # Normalize to unit vector
        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def compute_similarity(
        self,
        image: Union[np.ndarray, Image.Image],
        text: str,
    ) -> float:
        """
        Compute cosine similarity between an image and text.

        Args:
            image: BGR numpy array or PIL Image
            text: Text description

        Returns:
            Similarity score in [-1, 1]
        """
        img_embedding = self.embed_image(image)
        txt_embedding = self.embed_text(text)
        return float(np.dot(img_embedding, txt_embedding))
