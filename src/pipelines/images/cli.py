"""Command-line interface for image ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.pipelines.images import ImageIngestionPipeline
from src.pipelines.images.index import index_image_files, index_images_directory
from src.pipelines.images.models import ImageIngestionConfig

# The corrupt-image error message below (and any other message containing
# an em-dash) renders as "?" on an unconfigured Windows console — the same
# bug already found and fixed in ingest.py, smoke_test.py, and
# example_ingest.py this chapter; confirmed here too, live, while
# verifying the corrupt-image fix itself. Reconfigure both streams since
# this CLI writes JSON to stdout and errors to stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest image(s) with Tesseract OCR + OpenCLIP embeddings and index them into image_index."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write indexing summary JSON (e.g. {\"indexed\": count}) rather than chunk data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model architecture (default: ViT-B-32).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights tag (default: laion2b_s34b_b79k).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', 'mps'). Default: auto.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable Tesseract OCR extraction.",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable OpenCLIP embedding generation.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Path to tesseract binary executable.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.no_embedding:
        sys.stderr.write(
            "Error: Cannot index images into image_index with --no-embedding (embeddings are required for retrieval).\n"
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ImageIngestionConfig(
        ocr_enabled=not args.no_ocr,
        tesseract_cmd=args.tesseract_cmd,
        embedding_enabled=not args.no_embedding,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    pipeline = ImageIngestionPipeline(config=config)
    target = Path(args.input_path)

    if not target.exists():
        sys.stderr.write(f"Error: Path '{target}' does not exist.\n")
        sys.exit(1)

    if target.is_dir():
        # index_images_directory() calls the pipeline internally and hands
        # every ImageChunk with a non-None embedding to add_chunks() —
        # skipping unreadable files with a logged warning, same policy as
        # before.
        count = index_images_directory(target, pipeline=pipeline)
    else:
        # A single explicitly-named file — wrap in a list for index_image_files().
        # Validate that the file can be loaded so corrupt/non-image files are
        # caught and reported cleanly through the error handler instead of
        # silently swallowed by ingest_batch's skip policy.
        try:
            pipeline._load_image(target)
            count = index_image_files([target], pipeline=pipeline)
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            sys.exit(1)

    summary = {"indexed": count}

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Indexed {count} image(s) into image_index -> summary saved to {out_path}")
    else:
        print(f"Indexed {count} image(s) into image_index.")


if __name__ == "__main__":
    main()
