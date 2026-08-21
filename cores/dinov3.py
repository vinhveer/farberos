"""Small, production-friendly DINOv3 feature extraction wrapper.

The implementation uses the official DINOv3 checkpoints published on the
Hugging Face Hub.  Heavy dependencies are imported lazily so importing
``cores`` does not initialize PyTorch or download a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Sequence


ImageInput = Any  # PIL.Image, numpy array, torch tensor, or a local image path.


@dataclass(frozen=True)
class DINOv3Output:
    """Features returned by :meth:`DINOv3Inference.predict`.

    Attributes:
        embeddings: One global embedding per image, shaped ``[B, D]``.
        patch_embeddings: Dense patch tokens shaped ``[B, N, D]`` when
            requested and supported by the selected backbone.
    """

    embeddings: Any
    patch_embeddings: Any | None = None


class DINOv3Inference:
    """Load a DINOv3 backbone and extract global or dense image features.

    Args:
        model_name: Hugging Face model id or a local model directory.
        device: ``"cuda"``, ``"mps"``, ``"cpu"`` or ``"auto"``.
        dtype: Optional torch dtype. When omitted, float32 is used for stable
            feature extraction and visualization on every device.
        normalize: L2-normalize output features by default.
        local_files_only: Never contact the Hub; useful for offline inference.

    The default ViT-H+/16 checkpoint is the largest practical backbone for a
    single 16 GB inference GPU such as NVIDIA T4.
    """

    DEFAULT_MODEL = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

    def __init__(
        self,
        model_name: str | PathLike[str] = DEFAULT_MODEL,
        *,
        device: str = "auto",
        dtype: Any | None = None,
        normalize: bool = True,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv3Inference requires torch, transformers>=4.56 and Pillow. "
                "Install them with: pip install torch 'transformers>=4.56' pillow"
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(device)
        self.dtype = dtype or torch.float32
        self.normalize = normalize
        model_ref = str(model_name)

        self.processor = AutoImageProcessor.from_pretrained(
            model_ref, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_ref,
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> Any:
        torch = self._torch
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_image(image: ImageInput) -> ImageInput:
        if not isinstance(image, (str, PathLike)):
            return image
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Loading image paths requires Pillow") from exc

        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        # Copy detaches the returned image from the underlying file handle.
        with Image.open(path) as source:
            return source.convert("RGB").copy()

    def predict(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        image_size: int | None = None,
        return_patches: bool = False,
        normalize: bool | None = None,
        as_numpy: bool = False,
    ) -> DINOv3Output:
        """Extract DINOv3 features from one image or a batch.

        Input may be a PIL image, NumPy array, torch tensor, local path, or a
        sequence mixing those types. Outputs remain on the inference device
        unless ``as_numpy=True``.
        """

        torch = self._torch
        batch = list(images) if isinstance(images, (list, tuple)) else [images]
        if not batch:
            raise ValueError("images must contain at least one image")
        batch = [self._load_image(image) for image in batch]

        processor_kwargs: dict[str, Any] = {}
        if image_size is not None:
            target_size = {"height": image_size, "width": image_size}
            processor_kwargs.update(size=target_size, crop_size=target_size)
        inputs = self.processor(
            images=batch,
            return_tensors="pt",
            **processor_kwargs,
        )
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if torch.is_floating_point(value)
            else value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        embeddings = getattr(outputs, "pooler_output", None)
        hidden = getattr(outputs, "last_hidden_state", None)
        if embeddings is None:
            if hidden is None:
                raise RuntimeError("The selected model returned no usable image features")
            embeddings = hidden.mean(dim=1)

        patch_embeddings = self._get_patch_embeddings(hidden) if return_patches else None
        should_normalize = self.normalize if normalize is None else normalize
        if should_normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            if patch_embeddings is not None:
                patch_embeddings = torch.nn.functional.normalize(
                    patch_embeddings, p=2, dim=-1
                )

        if as_numpy:
            embeddings = embeddings.float().cpu().numpy()
            if patch_embeddings is not None:
                patch_embeddings = patch_embeddings.float().cpu().numpy()

        return DINOv3Output(
            embeddings=embeddings,
            patch_embeddings=patch_embeddings,
        )

    def _get_patch_embeddings(self, hidden: Any | None) -> Any | None:
        """Convert backbone-specific hidden states to ``[B, N, D]``."""
        if hidden is None:
            return None
        if hidden.ndim == 3:  # ViT: CLS, register tokens, then patch tokens.
            register_count = int(
                getattr(self.model.config, "num_register_tokens", 0) or 0
            )
            return hidden[:, 1 + register_count :]
        if hidden.ndim == 4:  # ConvNeXt: [B, C, H, W].
            return hidden.flatten(2).transpose(1, 2)
        raise RuntimeError(
            f"Unsupported last_hidden_state shape: {tuple(hidden.shape)}"
        )

    def __call__(
        self, images: ImageInput | Sequence[ImageInput], **kwargs: Any
    ) -> DINOv3Output:
        return self.predict(images, **kwargs)
