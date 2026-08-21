"""Farberos command-line entry point."""

from __future__ import annotations

import argparse

from cmds import ExtractDINOv3Command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trích xuất đặc trưng ảnh bằng DINOv3"
    )
    ExtractDINOv3Command.add_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return ExtractDINOv3Command.run(args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Lỗi: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
