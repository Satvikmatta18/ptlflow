#!/usr/bin/env python3
# Not a shell script: save as this file and run `python run_video_eval_suite.py ...` (do not paste into bash).
"""One-shot video evaluation: image metrics + optical-flow divergence + optional action-flow score.

Loads the optical flow model once. Reuses computed flow on the generated video for the
optional action-flow score when lengths align.

Typical usage:

    python run_video_eval_suite.py \\
        --gt_video assets/camera/01.mp4 \\
        --gen_video assets/camera/01_wangame.mp4 \\
        --output_dir outputs/eval_run_01

    python run_video_eval_suite.py ... --fp16 --max_frames 90

    python run_video_eval_suite.py ... \\
        --action_file assets/camera/01_action.npy --calibration calibration.json

Optional dependencies: scikit-image (SSIM), lpips (LPIPS).

Outputs: summary.json, image_metrics.json, metrics.csv, summary_flow.json,
         timeseries.png, comparison.mp4 (unless --no_viz).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from eval_flow_divergence import (
    compute_flow_sequence,
    compute_frame_metrics,
    compute_temporal_metrics,
    extract_frames,
    generate_visualizations,
)
from ptlflow.utils.io_adapter import IOAdapter


def _try_import_skimage_ssim():
    try:
        from skimage.metrics import structural_similarity

        return structural_similarity
    except ImportError:
        return None


def _try_import_lpips():
    try:
        import lpips

        return lpips
    except ImportError:
        return None


def _resize_frames_to_reference(
    frames: List[np.ndarray], target_hw: Tuple[int, int]
) -> List[np.ndarray]:
    th, tw = target_hw
    out = []
    for f in frames:
        if f.shape[0] == th and f.shape[1] == tw:
            out.append(f)
        else:
            out.append(cv.resize(f, (tw, th), interpolation=cv.INTER_AREA))
    return out


def _psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def compute_image_quality_metrics(
    frames_gt: List[np.ndarray],
    frames_gen: List[np.ndarray],
    device: torch.device,
    run_lpips: bool,
) -> Dict[str, Any]:
    """PSNR (always), SSIM / temporal SSIM if skimage, LPIPS if lpips."""
    n = min(len(frames_gt), len(frames_gen))
    psnr_list = []
    ssim_list = []
    temporal_ssim_list = []

    ssim_fn = _try_import_skimage_ssim()
    lpips_mod = _try_import_lpips() if run_lpips else None
    loss_fn = None
    if lpips_mod is not None:
        loss_fn = lpips_mod.LPIPS(net="alex").to(device)
        loss_fn.eval()

    for i in tqdm(range(n), desc="Image metrics", leave=False):
        g = frames_gt[i]
        p = frames_gen[i]
        psnr_list.append(_psnr_uint8(g, p))

        if ssim_fn is not None:
            rg = cv.cvtColor(g, cv.COLOR_BGR2RGB)
            rp = cv.cvtColor(p, cv.COLOR_BGR2RGB)
            ssim_list.append(
                float(ssim_fn(rg, rp, data_range=255, channel_axis=2))
            )

    if ssim_fn is not None and n >= 2:
        for i in tqdm(range(n - 1), desc="Temporal SSIM (gen)", leave=False):
            r0 = cv.cvtColor(frames_gen[i], cv.COLOR_BGR2RGB)
            r1 = cv.cvtColor(frames_gen[i + 1], cv.COLOR_BGR2RGB)
            temporal_ssim_list.append(
                float(ssim_fn(r0, r1, data_range=255, channel_axis=2))
            )

    lpips_list: List[float] = []
    if loss_fn is not None:
        for i in tqdm(range(n), desc="LPIPS", leave=False):
            g = frames_gt[i]
            p = frames_gen[i]
            t0 = torch.from_numpy(cv.cvtColor(g, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            t1 = torch.from_numpy(cv.cvtColor(p, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            t0 = t0.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            t1 = t1.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            with torch.no_grad():
                d = loss_fn(t0, t1)
            lpips_list.append(float(d.item()))

    result: Dict[str, Any] = {
        "n_frames_compared": n,
        "psnr_per_frame": psnr_list,
        "psnr_mean_db": float(np.mean(psnr_list)) if psnr_list else None,
    }
    if ssim_list:
        result["ssim_per_frame"] = ssim_list
        result["ssim_mean"] = float(np.mean(ssim_list))
    else:
        result["ssim_mean"] = None
        result["ssim_note"] = "install scikit-image for SSIM"

    if temporal_ssim_list:
        result["temporal_ssim_per_pair"] = temporal_ssim_list
        result["temporal_ssim_mean"] = float(np.mean(temporal_ssim_list))
    else:
        result["temporal_ssim_mean"] = None

    if lpips_list:
        result["lpips_per_frame"] = lpips_list
        result["lpips_mean"] = float(np.mean(lpips_list))
    elif run_lpips:
        result["lpips_mean"] = None
        result["lpips_note"] = "install lpips for LPIPS"
    else:
        result["lpips_mean"] = None

    return result


@torch.no_grad()
def _load_flow_model(model_name: str, ckpt: str, fp16: bool) -> torch.nn.Module:
    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if fp16:
            model = model.half()
    return model


def _evaluate_flow_divergence_from_frames(
    frames_gt: List[np.ndarray],
    frames_gen: List[np.ndarray],
    gt_video: str,
    gen_video: str,
    output_path: Path,
    model: torch.nn.Module,
    grid_size: int,
    fp16: bool,
    no_viz: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, float]], List[np.ndarray], List[np.ndarray]]:
    io_adapter = IOAdapter(
        output_stride=model.output_stride,
        input_size=frames_gt[0].shape[:2],
        cuda=torch.cuda.is_available(),
        fp16=fp16,
    )

    print("Computing GT flows...")
    flows_gt = compute_flow_sequence(model, frames_gt, io_adapter)
    print("Computing generated flows...")
    flows_gen = compute_flow_sequence(model, frames_gen, io_adapter)

    print("Computing flow divergence metrics...")
    frame_metrics_list: List[Dict[str, float]] = []
    for i in tqdm(range(len(flows_gt)), desc="Flow metrics", leave=False):
        m = compute_frame_metrics(flows_gt[i], flows_gen[i], grid_size=grid_size)
        m["frame_idx"] = i
        frame_metrics_list.append(m)

    summary = compute_temporal_metrics(frame_metrics_list)
    summary["gt_video"] = gt_video
    summary["gen_video"] = gen_video

    csv_path = output_path / "metrics.csv"
    fieldnames = list(frame_metrics_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics_list)
    print(f"  Wrote {csv_path}")

    flow_json = output_path / "summary_flow.json"
    with open(flow_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote {flow_json}")

    cap_tmp = cv.VideoCapture(gt_video)
    src_fps = cap_tmp.get(cv.CAP_PROP_FPS)
    cap_tmp.release()
    if src_fps <= 0:
        src_fps = 10.0

    if not no_viz:
        print("Generating flow visualizations...")
        generate_visualizations(
            frames_gt,
            frames_gen,
            flows_gt,
            flows_gen,
            frame_metrics_list,
            summary,
            output_path,
            fps=src_fps,
        )

    return summary, frame_metrics_list, flows_gt, flows_gen


def _action_flow_score_from_flows(
    flows_actual: List[np.ndarray],
    action_path: str,
    calibration_path: str,
    frame_shape: Tuple[int, int],
) -> float:
    """Mean pixel EPE between synthetic flow from actions and precomputed video flow."""
    from action_flow_score import SyntheticFlowGenerator, compute_pixel_epe

    actions = np.load(action_path, allow_pickle=True).item()
    keyboard = actions["keyboard"]
    mouse = actions["mouse"]
    n_actions = len(keyboard)
    n_flows_avail = len(flows_actual)
    n_pairs_from_actions = max(0, n_actions - 1)
    n_flows = min(n_flows_avail, n_pairs_from_actions)
    if n_flows < 1:
        raise ValueError("Not enough frames/actions for action-flow score (need >= 2 aligned steps)")

    if n_flows < n_flows_avail or n_flows < n_pairs_from_actions:
        print(
            f"  Warning: action-flow uses {n_flows} pairs "
            f"(video flows={n_flows_avail}, action pairs={n_pairs_from_actions})"
        )

    gen_cal = json.loads(Path(calibration_path).read_text())
    flow_gen_model = SyntheticFlowGenerator(gen_cal, frame_shape)

    epe_sum = 0.0
    for i in range(n_flows):
        flow_synth = flow_gen_model.generate_flow(keyboard[i], mouse[i])
        epe_sum += compute_pixel_epe(flow_synth, flows_actual[i])
    return epe_sum / n_flows


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading videos...\n  GT:  {args.gt_video}\n  Gen: {args.gen_video}")
    frames_gt = extract_frames(args.gt_video)
    frames_gen = extract_frames(args.gen_video)

    n = min(len(frames_gt), len(frames_gen))
    if len(frames_gt) != len(frames_gen):
        print(
            f"Warning: frame count mismatch (GT={len(frames_gt)}, Gen={len(frames_gen)}), "
            f"using first {n}"
        )
    frames_gt = frames_gt[:n]
    frames_gen = frames_gen[:n]

    if args.max_frames is not None:
        cap = min(n, args.max_frames)
        frames_gt = frames_gt[:cap]
        frames_gen = frames_gen[:cap]
        n = cap

    if n < 2:
        print("Error: need at least 2 frames for flow metrics.", file=sys.stderr)
        sys.exit(1)

    if args.resize_gen_to_gt:
        frames_gen = _resize_frames_to_reference(frames_gen, frames_gt[0].shape[:2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    merged: Dict[str, Any] = {
        "gt_video": args.gt_video,
        "gen_video": args.gen_video,
        "n_frames": n,
        "flow_model": args.model,
        "flow_ckpt": args.ckpt,
        "fp16": args.fp16,
    }

    if not args.no_image_metrics:
        print("--- Image quality metrics ---")
        img_metrics = compute_image_quality_metrics(
            frames_gt,
            frames_gen,
            device=device,
            run_lpips=not args.no_lpips,
        )
        merged["image"] = img_metrics
        with open(out / "image_metrics.json", "w") as f:
            json.dump(img_metrics, f, indent=2)
        print(f"  PSNR mean: {img_metrics.get('psnr_mean_db')}")
        if img_metrics.get("ssim_mean") is not None:
            print(f"  SSIM mean: {img_metrics['ssim_mean']:.4f}")
        if img_metrics.get("temporal_ssim_mean") is not None:
            print(f"  Temporal SSIM mean (gen): {img_metrics['temporal_ssim_mean']:.4f}")
        if img_metrics.get("lpips_mean") is not None:
            print(f"  LPIPS mean: {img_metrics['lpips_mean']:.4f}")

    print("--- Optical flow (single model load) ---")
    model = _load_flow_model(args.model, args.ckpt, args.fp16)

    flow_summary, _, _, flows_gen = _evaluate_flow_divergence_from_frames(
        frames_gt,
        frames_gen,
        args.gt_video,
        args.gen_video,
        out,
        model=model,
        grid_size=args.grid_size,
        fp16=args.fp16,
        no_viz=args.no_viz,
    )
    merged["flow_divergence"] = {k: v for k, v in flow_summary.items() if not k.startswith("_")}

    if args.action_file and args.calibration:
        print("--- Action-flow score (synthetic vs video flow, reusing gen flow) ---")
        try:
            afs = _action_flow_score_from_flows(
                flows_gen,
                args.action_file,
                args.calibration,
                frames_gen[0].shape[:2],
            )
            merged["action_flow_score"] = afs
            merged["action_file"] = args.action_file
            merged["calibration"] = args.calibration
            print(f"  action_flow_score (mean pixel EPE vs synthetic): {afs:.4f} (lower is better)")
        except ValueError as e:
            print(f"  Skipped action-flow score: {e}", file=sys.stderr)

    merged["elapsed_sec"] = round(time.time() - t0, 3)

    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nWrote {summary_path}")

    print("\n--- Flow summary (selected) ---")
    for key in [
        "pixel_epe_mean_mean",
        "fl_all_mean",
        "mf_angle_err_mean",
        "mf_cosine_mean",
        "flow_kl_2d_mean",
    ]:
        if key in flow_summary:
            v = flow_summary[key]
            if "fl_all" in key:
                print(f"  {key}: {v * 100:.2f}%")
            elif "angle" in key:
                print(f"  {key}: {v:.2f} deg")
            else:
                print(f"  {key}: {v:.4f}")
    div = flow_summary.get("divergence_onset_frame")
    print(f"  divergence_onset_frame: {div}")

    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run image + flow (+ optional action-flow) metrics in one pass.",
    )
    p.add_argument("--gt_video", type=str, required=True, help="Ground-truth video path")
    p.add_argument("--gen_video", type=str, required=True, help="Generated video path")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Use only the first N frames (after alignment)",
    )
    p.add_argument(
        "--resize_gen_to_gt",
        action="store_true",
        default=True,
        help="Resize generated frames to GT resolution (default: on)",
    )
    p.add_argument(
        "--no_resize_gen_to_gt",
        action="store_false",
        dest="resize_gen_to_gt",
        help="Disable resizing; videos must match in resolution",
    )
    p.add_argument("--model", type=str, default="dpflow", help="ptlflow model name")
    p.add_argument("--ckpt", type=str, default="things", help="ptlflow checkpoint id/path")
    p.add_argument("--fp16", action="store_true", help="Run flow model in half precision on CUDA")
    p.add_argument("--grid_size", type=int, default=8, help="Grid size for flow frame metrics")
    p.add_argument("--no_viz", action="store_true", help="Skip timeseries.png and comparison.mp4")
    p.add_argument("--no_image_metrics", action="store_true", help="Skip PSNR/SSIM/LPIPS")
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS even if lpips is installed")
    p.add_argument(
        "--action_file",
        type=str,
        default=None,
        help="Optional .npy with keyboard/mouse (see action_flow_score.py)",
    )
    p.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="calibration.json (required if --action_file is set)",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.action_file and not args.calibration:
        print("Error: --calibration is required when using --action_file", file=sys.stderr)
        sys.exit(1)
    run_suite(args)


if __name__ == "__main__":
    main()
