"""Run flow divergence evaluation on ALL video pairs and save results as .npy files.

Supports two modes:
    1. GT mode (default): Compare GT video flow vs generated video flow.
    2. Synthetic mode (--synthetic): Compare action-based synthetic flow vs generated
       video flow. No GT video needed.

Output structure mirrors assets/:
    outputs/flow_eval_all/{scenario}/{video_id}.npy

Each .npy file contains a dict with:
    - "summary": dict of summary statistics
    - "frame_metrics": list of per-frame metric dicts (the time series data)
    - "metric_names": list of metric names
    - "n_frames": number of flow frames
    - "gt_video" or "gen_video": paths to source videos
    - "mode": "gt" or "synthetic"

Usage:
    python run_all_eval.py
    python run_all_eval.py --output_dir outputs/flow_eval_all
    python run_all_eval.py --viz   # also generate comparison videos

    # Synthetic mode
    python run_all_eval.py --synthetic --calibration calibration.json
    python run_all_eval.py --synthetic --calibration calibration.json --no_depth
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter

from eval_flow_divergence import (
    extract_frames,
    compute_flow_sequence,
    compute_frame_metrics,
    compute_temporal_metrics,
    generate_visualizations,
)

import cv2 as cv


def run_gt_mode(args, assets_root, output_root):
    """Run evaluation in GT mode (original behavior)."""
    # Discover all video pairs
    pairs = []
    for scenario_dir in sorted(assets_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        if args.scenarios and scenario not in args.scenarios:
            continue
        for gt_video in sorted(scenario_dir.glob("*.mp4")):
            if "_wangame" in gt_video.name:
                continue
            video_id = gt_video.stem  # e.g. "01"
            gen_video = scenario_dir / f"{video_id}_wangame.mp4"
            if gen_video.exists():
                pairs.append((scenario, video_id, str(gt_video), str(gen_video)))

    print(f"Found {len(pairs)} video pairs across {len(set(p[0] for p in pairs))} scenarios")

    # Load model once
    print(f"Loading model: {args.model} ({args.ckpt})")
    model = ptlflow.get_model(args.model, args.ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    io_adapter = None
    total_time = 0
    results_summary = []

    for idx, (scenario, video_id, gt_path, gen_path) in enumerate(pairs):
        print(f"\n[{idx + 1}/{len(pairs)}] {scenario}/{video_id}")
        t0 = time.time()

        frames_gt = extract_frames(gt_path)
        frames_gen = extract_frames(gen_path)
        n_frames = min(len(frames_gt), len(frames_gen))
        if len(frames_gt) != len(frames_gen):
            print(f"  Warning: frame count mismatch (GT={len(frames_gt)}, Gen={len(frames_gen)})")
            frames_gt = frames_gt[:n_frames]
            frames_gen = frames_gen[:n_frames]

        if n_frames < 2:
            print("  Skipping: need at least 2 frames")
            continue

        shape = frames_gt[0].shape[:2]
        if io_adapter is None:
            io_adapter = IOAdapter(
                output_stride=model.output_stride,
                input_size=shape,
                cuda=torch.cuda.is_available(),
            )

        with torch.no_grad():
            flows_gt = compute_flow_sequence(model, frames_gt, io_adapter)
            flows_gen = compute_flow_sequence(model, frames_gen, io_adapter)

        frame_metrics_list = []
        for i in range(len(flows_gt)):
            m = compute_frame_metrics(flows_gt[i], flows_gen[i], grid_size=args.grid_size)
            m["frame_idx"] = i
            frame_metrics_list.append(m)

        summary = compute_temporal_metrics(frame_metrics_list)
        summary["gt_video"] = gt_path
        summary["gen_video"] = gen_path
        summary["mode"] = "gt"

        out_dir = output_root / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{video_id}.npy"

        save_data = {
            "summary": summary,
            "frame_metrics": frame_metrics_list,
            "metric_names": list(frame_metrics_list[0].keys()),
            "n_frames": len(flows_gt),
            "gt_video": gt_path,
            "gen_video": gen_path,
            "mode": "gt",
        }
        np.save(str(npy_path), save_data, allow_pickle=True)

        if args.viz:
            viz_dir = out_dir / video_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            cap_tmp = cv.VideoCapture(gt_path)
            fps = cap_tmp.get(cv.CAP_PROP_FPS)
            cap_tmp.release()
            if fps <= 0:
                fps = 10.0
            generate_visualizations(
                frames_gt, frames_gen, flows_gt, flows_gen,
                frame_metrics_list, summary, viz_dir, fps=fps,
            )

        elapsed = time.time() - t0
        total_time += elapsed

        print(f"  {n_frames} frames, {elapsed:.1f}s")
        print(f"  pixel_epe={summary.get('pixel_epe_mean_mean', 0):.2f}  "
              f"fl_all={summary.get('fl_all_mean', 0) * 100:.1f}%  "
              f"mf_angle={summary.get('mf_angle_err_mean', 0):.1f}\u00b0  "
              f"foe_dist={summary.get('foe_dist_mean', 0):.0f}  "
              f"kl_2d={summary.get('flow_kl_2d_mean', 0):.3f}")

        results_summary.append({
            "scenario": scenario,
            "video_id": video_id,
            "pixel_epe_mean": summary.get("pixel_epe_mean_mean", 0),
            "fl_all": summary.get("fl_all_mean", 0),
            "mf_angle_err": summary.get("mf_angle_err_mean", 0),
            "mf_cosine": summary.get("mf_cosine_mean", 0),
            "foe_dist": summary.get("foe_dist_mean", 0),
            "flow_kl_2d": summary.get("flow_kl_2d_mean", 0),
        })

    return results_summary, total_time


def run_synthetic_mode(args, assets_root, output_root):
    """Run evaluation in synthetic mode (action-based flow vs generated video flow)."""
    from synthetic_flow import SyntheticFlowGenerator, load_calibration
    from eval_flow_divergence import generate_visualizations_synthetic

    # Load calibration
    calibration = load_calibration(args.calibration)
    print(f"Loaded calibration from {args.calibration}")
    use_depth = not args.no_depth

    # Discover all gen video + action pairs
    pairs = []
    for scenario_dir in sorted(assets_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        if args.scenarios and scenario not in args.scenarios:
            continue
        for gen_video in sorted(scenario_dir.glob("*_wangame.mp4")):
            video_id = gen_video.name.replace("_wangame.mp4", "")
            action_path = scenario_dir / f"{video_id}_action.npy"
            if action_path.exists():
                pairs.append((scenario, video_id, str(gen_video), str(action_path)))

    print(f"Found {len(pairs)} gen+action pairs across {len(set(p[0] for p in pairs))} scenarios")

    # Load flow model once
    print(f"Loading flow model: {args.model} ({args.ckpt})")
    model = ptlflow.get_model(args.model, args.ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    io_adapter = None
    flow_generator = None
    total_time = 0
    results_summary = []

    for idx, (scenario, video_id, gen_path, action_path) in enumerate(pairs):
        print(f"\n[{idx + 1}/{len(pairs)}] {scenario}/{video_id}")
        t0 = time.time()

        # Extract gen frames
        frames_gen = extract_frames(gen_path)
        n_frames = len(frames_gen)

        # Load actions
        actions = np.load(action_path, allow_pickle=True).item()
        n_actions = len(actions["keyboard"])
        n_use = min(n_frames, n_actions)
        if n_frames != n_actions:
            print(f"  Warning: frame/action mismatch ({n_frames} vs {n_actions})")
            frames_gen = frames_gen[:n_use]
            n_frames = n_use

        if n_frames < 2:
            print("  Skipping: need at least 2 frames")
            continue

        shape = frames_gen[0].shape[:2]
        if io_adapter is None:
            io_adapter = IOAdapter(
                output_stride=model.output_stride,
                input_size=shape,
                cuda=torch.cuda.is_available(),
            )

        # Initialize flow generator once
        if flow_generator is None:
            flow_generator = SyntheticFlowGenerator(
                calibration=calibration,
                frame_shape=shape,
                use_depth=use_depth,
            )
            if use_depth:
                print("Loading depth model...")
                flow_generator.load_depth_model()

        # Compute gen video flows
        with torch.no_grad():
            flows_gen = compute_flow_sequence(model, frames_gen, io_adapter)

        # Generate synthetic flows (with depth maps for visualization)
        flows_synth, depth_maps = flow_generator.generate_flow_sequence(
            actions=actions,
            frames=frames_gen if use_depth else None,
            return_depths=True,
        )

        # Match lengths
        n_flows = min(len(flows_gen), len(flows_synth))
        flows_gen = flows_gen[:n_flows]
        flows_synth = flows_synth[:n_flows]

        # Compute metrics
        frame_metrics_list = []
        for i in range(n_flows):
            m = compute_frame_metrics(flows_synth[i], flows_gen[i], grid_size=args.grid_size)
            m["frame_idx"] = i
            frame_metrics_list.append(m)

        summary = compute_temporal_metrics(frame_metrics_list)
        summary["gen_video"] = gen_path
        summary["action_file"] = action_path
        summary["mode"] = "synthetic"

        # Save as .npy
        out_dir = output_root / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{video_id}.npy"

        save_data = {
            "summary": summary,
            "frame_metrics": frame_metrics_list,
            "metric_names": list(frame_metrics_list[0].keys()),
            "n_frames": n_flows,
            "gen_video": gen_path,
            "action_file": action_path,
            "mode": "synthetic",
        }
        np.save(str(npy_path), save_data, allow_pickle=True)

        # Optional: generate comparison video
        if args.viz:
            viz_dir = out_dir / video_id
            viz_dir.mkdir(parents=True, exist_ok=True)
            cap_tmp = cv.VideoCapture(gen_path)
            fps = cap_tmp.get(cv.CAP_PROP_FPS)
            cap_tmp.release()
            if fps <= 0:
                fps = 10.0

            # Load GT frames for reference if available
            gt_video_path = assets_root / scenario / f"{video_id}.mp4"
            frames_gt = None
            if gt_video_path.exists():
                frames_gt = extract_frames(str(gt_video_path))
                frames_gt = frames_gt[:n_frames]

            generate_visualizations_synthetic(
                frames_gen, flows_synth, flows_gen,
                frame_metrics_list, summary, viz_dir, fps=fps,
                actions=actions, depth_maps=depth_maps, frames_gt=frames_gt,
            )

        elapsed = time.time() - t0
        total_time += elapsed

        print(f"  {n_frames} frames, {elapsed:.1f}s")
        print(f"  pixel_epe={summary.get('pixel_epe_mean_mean', 0):.2f}  "
              f"fl_all={summary.get('fl_all_mean', 0) * 100:.1f}%  "
              f"mf_angle={summary.get('mf_angle_err_mean', 0):.1f}\u00b0  "
              f"foe_dist={summary.get('foe_dist_mean', 0):.0f}  "
              f"kl_2d={summary.get('flow_kl_2d_mean', 0):.3f}")

        results_summary.append({
            "scenario": scenario,
            "video_id": video_id,
            "pixel_epe_mean": summary.get("pixel_epe_mean_mean", 0),
            "fl_all": summary.get("fl_all_mean", 0),
            "mf_angle_err": summary.get("mf_angle_err_mean", 0),
            "mf_cosine": summary.get("mf_cosine_mean", 0),
            "foe_dist": summary.get("foe_dist_mean", 0),
            "flow_kl_2d": summary.get("flow_kl_2d_mean", 0),
        })

    return results_summary, total_time


def print_results_table(results_summary, total_time, output_root, mode="gt"):
    """Print summary table of results."""
    np.save(str(output_root / "all_summaries.npy"), results_summary, allow_pickle=True)

    mode_label = "GT" if mode == "gt" else "Synthetic"
    print(f"\n{'=' * 80}")
    print(f"[{mode_label} Mode] Completed {len(results_summary)} pairs in {total_time:.0f}s "
          f"({total_time / max(len(results_summary), 1):.1f}s/pair)")
    print(f"{'=' * 80}")
    print(f"{'Scenario':<45} {'EPE':>6} {'FL%':>6} {'Angle':>6} {'FOE':>7} {'KL':>7}")
    print("-" * 80)
    for r in results_summary:
        name = f"{r['scenario']}/{r['video_id']}"
        print(f"{name:<45} {r['pixel_epe_mean']:>6.2f} {r['fl_all'] * 100:>5.1f}% "
              f"{r['mf_angle_err']:>5.1f}\u00b0 {r['foe_dist']:>7.0f} {r['flow_kl_2d']:>7.3f}")

    # Per-scenario averages
    print(f"\n{'Per-scenario averages':}")
    print(f"{'Scenario':<45} {'EPE':>6} {'FL%':>6} {'Angle':>6} {'FOE':>7} {'KL':>7}")
    print("-" * 80)
    scenarios = sorted(set(r["scenario"] for r in results_summary))
    for sc in scenarios:
        sc_results = [r for r in results_summary if r["scenario"] == sc]
        n = len(sc_results)
        print(f"{sc:<45} "
              f"{sum(r['pixel_epe_mean'] for r in sc_results) / n:>6.2f} "
              f"{sum(r['fl_all'] for r in sc_results) / n * 100:>5.1f}% "
              f"{sum(r['mf_angle_err'] for r in sc_results) / n:>5.1f}\u00b0 "
              f"{sum(r['foe_dist'] for r in sc_results) / n:>7.0f} "
              f"{sum(r['flow_kl_2d'] for r in sc_results) / n:>7.3f}")

    print(f"\nResults saved to: {output_root}/")
    print(f"  Per-video .npy: {output_root}/{{scenario}}/{{video_id}}.npy")
    print(f"  All summaries:  {output_root}/all_summaries.npy")


def main():
    parser = argparse.ArgumentParser(description="Batch flow eval on all video pairs")
    parser.add_argument("--assets_dir", type=str, default="assets",
                        help="Root assets directory")
    parser.add_argument("--output_dir", type=str, default="outputs/flow_eval_all",
                        help="Output directory for .npy files")
    parser.add_argument("--model", type=str, default="dpflow",
                        help="Optical flow model")
    parser.add_argument("--ckpt", type=str, default="things",
                        help="Model checkpoint")
    parser.add_argument("--viz", action="store_true",
                        help="Also generate comparison videos")
    parser.add_argument("--grid_size", type=int, default=8)

    # Synthetic mode options
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic flow from actions instead of GT video flow")
    parser.add_argument("--calibration", type=str,
                        help="Path to calibration.json (required with --synthetic)")
    parser.add_argument("--no_depth", action="store_true",
                        help="Skip depth estimation, use constant depth")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Only evaluate these scenarios (e.g. fully_random camera)")
    args = parser.parse_args()

    assets_root = Path(args.assets_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        if not args.calibration:
            print("Error: --calibration is required with --synthetic")
            sys.exit(1)
        results_summary, total_time = run_synthetic_mode(args, assets_root, output_root)
        mode = "synthetic"
    else:
        results_summary, total_time = run_gt_mode(args, assets_root, output_root)
        mode = "gt"

    if results_summary:
        print_results_table(results_summary, total_time, output_root, mode=mode)


if __name__ == "__main__":
    main()
