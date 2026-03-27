#!/usr/bin/env python3
"""Generate with FastVideo MatrixGame 2.0 and score against ground truth.

This is an end-to-end wrapper around two existing stages:
1. FastVideo generation using MatrixGame 2.0
2. `run_video_eval_suite.py` scoring on the generated MP4

Typical usage:

    python run_matrixgame2_benchmark.py \
        --model_path FastVideo/Matrix-Game-2.0-Base-Diffusers \
        --gt_video to_shao/camera/01.mp4 \
        --action_file to_shao/camera/01_action.npy \
        --conditioning_image to_shao/camera/01.jpg \
        --output_dir outputs/matrixgame2/camera_01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_video_eval_suite import build_parser as build_eval_parser
from run_video_eval_suite import run_suite


def _add_fastvideo_to_path(repo_root: Path) -> None:
    fastvideo_root = repo_root / "FastVideo"
    if not fastvideo_root.exists():
        raise FileNotFoundError(f"FastVideo checkout not found at {fastvideo_root}")
    fastvideo_root_str = str(fastvideo_root)
    if fastvideo_root_str not in sys.path:
        sys.path.insert(0, fastvideo_root_str)


def _default_conditioning_image(gt_video: Path) -> Path:
    candidate = gt_video.with_suffix(".jpg")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "No conditioning image provided and no sibling .jpg was found for "
        f"{gt_video}"
    )


def _load_action_controls(action_file: Path, num_frames: int) -> dict[str, torch.Tensor]:
    action_data = np.load(action_file, allow_pickle=True)
    if isinstance(action_data, np.ndarray) and action_data.dtype == object:
        action_data = action_data.item()

    if isinstance(action_data, dict):
        keyboard = np.asarray(action_data["keyboard"][:num_frames], dtype=np.float32)
        mouse = np.asarray(action_data["mouse"][:num_frames], dtype=np.float32)
        return {
            "keyboard_cond": torch.from_numpy(keyboard).unsqueeze(0),
            "mouse_cond": torch.from_numpy(mouse).unsqueeze(0),
        }

    keyboard = np.asarray(action_data[:num_frames], dtype=np.float32)
    return {
        "keyboard_cond": torch.from_numpy(keyboard).unsqueeze(0),
    }


def _generate_with_fastvideo(args: argparse.Namespace, gen_video_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    _add_fastvideo_to_path(repo_root)

    from fastvideo import VideoGenerator

    conditioning_image = (
        Path(args.conditioning_image)
        if args.conditioning_image is not None
        else _default_conditioning_image(Path(args.gt_video))
    )

    controls = _load_action_controls(Path(args.action_file), args.num_frames)

    generator = VideoGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        workload_type="i2v",
    )

    result = generator.generate_video(
        prompt=args.prompt,
        image_path=str(conditioning_image),
        output_path=str(gen_video_path),
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        save_video=True,
        return_frames=False,
        **controls,
    )

    video_path = result.get("video_path")
    if not video_path:
        raise RuntimeError("FastVideo generation completed without returning a saved video path")

    return {
        "model_path": args.model_path,
        "conditioning_image": str(conditioning_image),
        "generated_video": str(video_path),
        "generation_time": result.get("generation_time"),
        "peak_memory_mb": result.get("peak_memory_mb"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate with FastVideo MatrixGame 2.0, then run the eval suite.",
    )
    p.add_argument("--model_path", type=str, required=True, help="FastVideo model path or HF repo id")
    p.add_argument("--gt_video", type=str, required=True, help="Ground-truth video path")
    p.add_argument("--action_file", type=str, required=True, help="MatrixGame action .npy")
    p.add_argument(
        "--conditioning_image",
        type=str,
        default=None,
        help="Conditioning image for MatrixGame I2V. Defaults to sibling .jpg of --gt_video",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Benchmark output directory")
    p.add_argument("--prompt", type=str, default="", help="Prompt passed to FastVideo")
    p.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for FastVideo")
    p.add_argument("--seed", type=int, default=1024, help="FastVideo seed")
    p.add_argument("--num_inference_steps", type=int, default=3, help="FastVideo denoising steps")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="FastVideo guidance scale")
    p.add_argument("--num_frames", type=int, default=57, help="Generated frames")
    p.add_argument("--height", type=int, default=352, help="Generated height")
    p.add_argument("--width", type=int, default=640, help="Generated width")
    p.add_argument("--fps", type=int, default=25, help="Generated FPS")

    # Eval-stage args mirror the existing suite where it matters.
    p.add_argument("--max_frames", type=int, default=None, help="Eval only the first N aligned frames")
    p.add_argument("--no_resize_gen_to_gt", action="store_true", help="Disable resizing generated frames to GT")
    p.add_argument("--flow_model", type=str, default="dpflow", help="ptlflow model name")
    p.add_argument("--flow_ckpt", type=str, default="things", help="ptlflow checkpoint id/path")
    p.add_argument("--fp16", action="store_true", help="Run ptlflow in half precision on CUDA")
    p.add_argument("--grid_size", type=int, default=8, help="Grid size for flow metrics")
    p.add_argument("--no_viz", action="store_true", help="Skip flow visualizations")
    p.add_argument("--no_image_metrics", action="store_true", help="Skip PSNR/SSIM/LPIPS")
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS")
    p.add_argument(
        "--calibration",
        type=str,
        default="calibration.json",
        help="Calibration JSON for action-flow scoring",
    )
    return p


def _build_eval_args(args: argparse.Namespace, gen_video_path: Path, output_dir: Path) -> argparse.Namespace:
    eval_parser = build_eval_parser()
    eval_args = eval_parser.parse_args(
        [
            "--gt_video",
            args.gt_video,
            "--gen_video",
            str(gen_video_path),
            "--output_dir",
            str(output_dir),
            "--model",
            args.flow_model,
            "--ckpt",
            args.flow_ckpt,
            "--grid_size",
            str(args.grid_size),
            "--action_file",
            args.action_file,
            "--calibration",
            args.calibration,
        ]
    )
    eval_args.max_frames = args.max_frames
    eval_args.resize_gen_to_gt = not args.no_resize_gen_to_gt
    eval_args.fp16 = args.fp16
    eval_args.no_viz = args.no_viz
    eval_args.no_image_metrics = args.no_image_metrics
    eval_args.no_lpips = args.no_lpips
    return eval_args


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_video_path = output_dir / "generated.mp4"

    generation = _generate_with_fastvideo(args, gen_video_path)

    eval_args = _build_eval_args(args, gen_video_path, output_dir)
    summary = run_suite(eval_args)
    summary["generation"] = generation

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nUpdated {summary_path} with generation metadata")


if __name__ == "__main__":
    main()
