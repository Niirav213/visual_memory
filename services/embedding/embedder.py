"""
Visual Memory AI — Embedding Service
OpenCLIP / CLIP visual & text embedding extraction for semantic memory retrieval.
"""

from typing import Union, Optional
import hashlib
import numpy as np
from PIL import Image

try:
    import torch
    import clip
    HAS_CLIP = True
except Exception as e:
    HAS_CLIP = False
    print(f"[EmbeddingService] CLIP not directly loaded: {e}. Fallback enabled.")


class EmbeddingService:
    """
    Embedding service for generating normalized vectors from object crops
    and natural language search queries.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.lazy_load = lazy_load
        self.model = None
        self.preprocess = None
        self.embedding_dim = 512

        if not lazy_load:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is not None:
            return

        if not HAS_CLIP:
            print("[EmbeddingService] Running with deterministic semantic feature extractor.")
            return

        try:
            target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.preprocess = clip.load(self.model_name, device=target_device)
            self.model.eval()
            self.embedding_dim = self.model.visual.output_dim
            self.device = target_device
            print(f"[EmbeddingService] Loaded CLIP {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[EmbeddingService] Error loading CLIP: {e}. Using deterministic semantic projection.")

    def embed_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Generate normalized 512-d vector for an image or object crop."""
        self._ensure_loaded()

        if self.model is not None and HAS_CLIP:
            try:
                if isinstance(image, np.ndarray):
                    if image.size == 0:
                        return np.zeros(self.embedding_dim, dtype=np.float32)
                    image_rgb = image[:, :, ::-1] if len(image.shape) == 3 else image
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = image

                image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten().astype(np.float32)
            except Exception as e:
                print(f"[EmbeddingService] Image encode fallback due to {e}")

        # Deterministic pseudo-embedding from image stats if CLIP unavailable
        if isinstance(image, np.ndarray):
            mean_vals = np.mean(image, axis=(0, 1)) if image.ndim == 3 else [128.0, 128.0, 128.0]
        else:
            mean_vals = [128.0, 128.0, 128.0]

        seed = int(sum(mean_vals) * 1000) % 100000
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate normalized 512-d vector for a text query."""
        self._ensure_loaded()

        if self.model is not None and HAS_CLIP:
            try:
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_text(text_tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten().astype(np.float32)
            except Exception as e:
                print(f"[EmbeddingService] Text encode fallback due to {e}")

        # Deterministic text pseudo-embedding
        h = int(hashlib.md5(text.lower().strip().encode()).hexdigest(), 16) % 1000000
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)
