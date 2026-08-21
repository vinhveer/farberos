"""CLI command for extracting DINOv3 image features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cores import DINOv3Inference


class ExtractDINOv3Command:
    """Extract global DINOv3 embeddings from an image or image folder."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", type=Path, help="Path to an image or image folder")
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            help="Output .npy file, or output directory when INPUT is a folder",
        )
        parser.add_argument(
            "--model",
            default=DINOv3Inference.DEFAULT_MODEL,
            help="Hugging Face model id or local model directory",
        )
        parser.add_argument(
            "--device",
            default="auto",
            choices=("auto", "cpu", "cuda", "mps"),
            help="Inference device (default: auto)",
        )
        parser.add_argument(
            "--dtype",
            default="float32",
            choices=("float32", "float16"),
            help="Model precision; float32 is safer for feature maps (default: float32)",
        )
        parser.add_argument(
            "--no-normalize",
            action="store_true",
            help="Do not L2-normalize the embedding",
        )
        parser.add_argument(
            "--visualation",
            "--visualization",
            nargs="?",
            const=True,
            metavar="PATH",
            help=(
                "Save DINOv3 feature maps. PATH is a file for one image or "
                "a directory for folder input"
            ),
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> int:
        import torch

        source = args.input.expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Không tìm thấy input: {source}")

        inference = DINOv3Inference(
            model_name=args.model,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            normalize=not args.no_normalize,
        )
        if source.is_file():
            return cls._run_file(inference, source, args)

        images = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"Folder không có ảnh được hỗ trợ: {source}")

        feature_root = (
            args.output.expanduser()
            if args.output is not None
            else Path("outputs") / f"{source.name}-features"
        )
        map_root = None
        if args.visualation is not None:
            map_root = (
                Path(args.visualation).expanduser()
                if args.visualation is not True
                else Path("outputs") / f"{source.name}-maps"
            )

        for index, image in enumerate(images, start=1):
            relative = image.relative_to(source)
            feature_path = (feature_root / relative).with_suffix(".npy")
            map_path = (
                (map_root / relative).with_suffix(".jpg")
                if map_root is not None
                else None
            )
            print(f"[{index}/{len(images)}] {relative}", file=sys.stderr)
            cls._extract_and_save(
                inference,
                image,
                feature_path=feature_path,
                map_path=map_path,
            )

        print(f"Đã xử lý {len(images)} ảnh. Features: {feature_root}")
        if map_root is not None:
            print(f"Feature maps: {map_root}")
        return 0

    @classmethod
    def _run_file(
        cls,
        inference: DINOv3Inference,
        image: Path,
        args: argparse.Namespace,
    ) -> int:
        map_path = (
            cls._visualization_path(image, args.visualation)
            if args.visualation is not None
            else None
        )
        if args.output is not None:
            feature_path = args.output.expanduser()
            if feature_path.suffix.lower() != ".npy":
                feature_path = feature_path.with_suffix(".npy")
            feature = cls._extract_and_save(
                inference,
                image,
                feature_path=feature_path,
                map_path=map_path,
            )
            print(f"Đã lưu đặc trưng {feature.shape} vào: {feature_path}")
        else:
            feature = cls._extract_and_save(
                inference,
                image,
                feature_path=None,
                map_path=map_path,
            )
            print(
                json.dumps(
                    {
                        "image": str(image.resolve()),
                        "model": args.model,
                        "shape": list(feature.shape),
                        "embedding": feature.tolist(),
                    },
                    ensure_ascii=False,
                )
            )
        if map_path is not None:
            print(f"Đã lưu feature map vào: {map_path}", file=sys.stderr)
        return 0

    @classmethod
    def _extract_and_save(
        cls,
        inference: DINOv3Inference,
        image: Path,
        *,
        feature_path: Path | None,
        map_path: Path | None,
    ) -> object:
        result = inference.predict(
            image,
            return_patches=map_path is not None,
            as_numpy=True,
        )
        feature = result.embeddings[0]
        import numpy as np

        if not np.isfinite(feature).all():
            raise RuntimeError(
                "Model trả về embedding có NaN/Inf. Hãy chạy lại với "
                "--dtype float32; nếu vẫn lỗi, dùng model nhỏ hơn."
            )
        if map_path is not None and not np.isfinite(result.patch_embeddings).all():
            raise RuntimeError(
                "Model trả về patch features có NaN/Inf. Hãy chạy lại với "
                "--dtype float32; nếu vẫn lỗi, dùng model nhỏ hơn."
            )
        if feature_path is not None:
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, feature)
        if map_path is not None:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            cls._save_feature_map(
                image=image,
                output=map_path,
                global_feature=feature,
                patch_features=result.patch_embeddings[0],
            )
        return feature

    @staticmethod
    def _visualization_path(image: Path, value: bool | str) -> Path:
        if value is True:
            return image.with_name(f"{image.stem}_dinov3_map.jpg")
        output = Path(value).expanduser()
        if not output.suffix:
            output = output.with_suffix(".jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    @staticmethod
    def _save_feature_map(
        *,
        image: Path,
        output: Path,
        global_feature: object,
        patch_features: object,
    ) -> None:
        """Render cosine similarity between each patch and the global token."""
        import numpy as np
        from PIL import Image

        patches = np.asarray(patch_features, dtype=np.float32)
        global_vector = np.asarray(global_feature, dtype=np.float32)
        patches /= np.maximum(np.linalg.norm(patches, axis=-1, keepdims=True), 1e-12)
        global_vector /= max(float(np.linalg.norm(global_vector)), 1e-12)
        scores = patches @ global_vector

        patch_count = scores.shape[0]
        grid_height = int(round(patch_count**0.5))
        while grid_height > 1 and patch_count % grid_height:
            grid_height -= 1
        grid_width = patch_count // grid_height
        heat = scores.reshape(grid_height, grid_width)
        low, high = np.percentile(heat, [5, 95])
        heat = np.clip((heat - low) / max(high - low, 1e-6), 0.0, 1.0)

        # Compact blue -> cyan -> yellow -> red colour map.
        stops = np.array(
            [[20, 35, 120], [20, 190, 220], [250, 220, 40], [220, 35, 25]],
            dtype=np.float32,
        )
        position = heat * (len(stops) - 1)
        left = np.floor(position).astype(int)
        right = np.minimum(left + 1, len(stops) - 1)
        weight = (position - left)[..., None]
        colours = stops[left] * (1.0 - weight) + stops[right] * weight

        with Image.open(image) as source:
            original = source.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            heatmap = Image.fromarray(colours.astype(np.uint8), mode="RGB").resize(
                original.size, resample=resampling.BICUBIC
            )
            overlay = Image.blend(original, heatmap, alpha=0.45)
            overlay.save(output, quality=95)
