"""Calibrate synthetic flow parameters from ground truth videos.

Learns the mapping coefficients (alpha_yaw, alpha_pitch, alpha_turn, beta_fwd,
beta_strafe) by regressing GT optical flow against the physics model's basis
fields, using linear least squares across many frames and pixels.

Output: calibration.json with the learned coefficients.

Usage:
    python calibrate_flow.py --assets_dir assets --output calibration.json
    python calibrate_flow.py --assets_dir assets --output calibration.json \
        --scenarios 1_wasd_only camera camera4hold_alpha1
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import tensor_dict_to_numpy


def extract_frames(video_path: str) -> List[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


@torch.no_grad()
def compute_flow_pair(
    model: torch.nn.Module,
    io_adapter: IOAdapter,
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> np.ndarray:
    """Compute optical flow between two frames. Returns HxWx2."""
    inputs = io_adapter.prepare_inputs([frame1, frame2])
    preds = model(inputs)
    preds["images"] = inputs["images"]
    preds = io_adapter.unscale(preds)
    preds_npy = tensor_dict_to_numpy(preds)
    return preds_npy["flows"]


def build_regression_data(
    flows: List[np.ndarray],
    actions: dict,
    focal_length: float,
    n_pixels_per_frame: int = 500,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build design matrix and target vector for calibration regression.

    For each sampled pixel at position (x, y) relative to the image center,
    the predicted flow is linear in the calibration parameters:

        u_pred = alpha_yaw * ms[1] * (-(f + x^2/f))
               + alpha_pitch * ms[0] * (xy/f)
               + alpha_turn * (kb[5]-kb[4]) * (-(f + x^2/f))
               + beta_fwd * (kb[0]-kb[1]) * x
               + beta_strafe * (kb[3]-kb[2]) * (-f)

        v_pred = alpha_pitch * ms[0] * (f + y^2/f)
               + alpha_yaw * ms[1] * (-(xy/f))
               + alpha_turn * (kb[5]-kb[4]) * (-(xy/f))
               + beta_fwd * (kb[0]-kb[1]) * y
               + beta_strafe * 0   (strafe has no v-component at Z=1)

    Parameters
    ----------
    flows : list of HxWx2 arrays (GT optical flow)
    actions : dict with keyboard (T, 6) and mouse (T, 2)
    focal_length : float, assumed focal length in pixels
    n_pixels_per_frame : number of pixels to sample per frame
    rng : random number generator

    Returns
    -------
    A : np.ndarray, shape (N, 5) — design matrix
    b : np.ndarray, shape (N,) — target values
    Where N = 2 * n_flows * n_pixels_per_frame (u and v stacked)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    keyboard = actions["keyboard"]
    mouse_arr = actions["mouse"]
    n_flows = min(len(flows), len(keyboard) - 1)

    H, W = flows[0].shape[:2]
    cx, cy = W / 2.0, H / 2.0
    f = focal_length

    all_A_rows = []
    all_b_vals = []

    for i in range(n_flows):
        flow = flows[i]
        kb = keyboard[i]
        ms = mouse_arr[i]

        # Sample random pixels
        ys = rng.integers(0, H, size=n_pixels_per_frame)
        xs = rng.integers(0, W, size=n_pixels_per_frame)

        # Pixel coords relative to center
        x = xs.astype(np.float64) - cx
        y = ys.astype(np.float64) - cy

        # Pre-compute spatial terms
        xy_over_f = x * y / f
        f_plus_x2_over_f = f + x ** 2 / f
        f_plus_y2_over_f = f + y ** 2 / f

        # Action values (scalars)
        yaw_val = float(ms[1])
        pitch_val = float(ms[0])
        turn_val = float(kb[5] - kb[4])
        fwd_val = float(kb[0] - kb[1])
        strafe_val = float(kb[3] - kb[2])

        # --- U component rows ---
        # Columns: [alpha_yaw, alpha_pitch, alpha_turn, beta_fwd, beta_strafe]
        col_ayaw_u = yaw_val * (-f_plus_x2_over_f)
        col_apitch_u = pitch_val * xy_over_f
        col_aturn_u = turn_val * (-f_plus_x2_over_f)
        col_bfwd_u = fwd_val * x
        col_bstrafe_u = strafe_val * (-f) * np.ones(n_pixels_per_frame)

        A_u = np.column_stack([col_ayaw_u, col_apitch_u, col_aturn_u,
                               col_bfwd_u, col_bstrafe_u])
        b_u = flow[ys, xs, 0].astype(np.float64)  # actual u flow

        # --- V component rows ---
        col_ayaw_v = yaw_val * (-xy_over_f)
        col_apitch_v = pitch_val * f_plus_y2_over_f
        col_aturn_v = turn_val * (-xy_over_f)
        col_bfwd_v = fwd_val * y
        col_bstrafe_v = np.zeros(n_pixels_per_frame)  # strafe has no v-component

        A_v = np.column_stack([col_ayaw_v, col_apitch_v, col_aturn_v,
                               col_bfwd_v, col_bstrafe_v])
        b_v = flow[ys, xs, 1].astype(np.float64)  # actual v flow

        all_A_rows.append(A_u)
        all_A_rows.append(A_v)
        all_b_vals.append(b_u)
        all_b_vals.append(b_v)

    A = np.vstack(all_A_rows)
    b = np.concatenate(all_b_vals)
    return A, b


def calibrate(
    assets_dir: str,
    scenarios: List[str],
    model_name: str = "dpflow",
    ckpt: str = "things",
    focal_length: float = 457.0,
    n_pixels_per_frame: int = 500,
) -> dict:
    """Run calibration on specified scenarios.

    Returns dict of calibration coefficients.
    """
    assets_root = Path(assets_dir)
    rng = np.random.default_rng(42)

    # Load flow model
    print(f"Loading flow model: {model_name} ({ckpt})")
    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    io_adapter = None
    all_A = []
    all_b = []
    frame_shape = None

    for scenario in scenarios:
        scenario_dir = assets_root / scenario
        if not scenario_dir.exists():
            print(f"  Warning: scenario dir not found: {scenario_dir}")
            continue

        gt_videos = sorted(scenario_dir.glob("*.mp4"))
        gt_videos = [v for v in gt_videos if "_wangame" not in v.name]

        for gt_video in gt_videos:
            video_id = gt_video.stem
            action_path = scenario_dir / f"{video_id}_action.npy"
            if not action_path.exists():
                continue

            print(f"  Processing {scenario}/{video_id}...")

            # Extract frames
            frames = extract_frames(str(gt_video))
            if len(frames) < 2:
                continue

            # Create IO adapter if needed
            if io_adapter is None:
                frame_shape = frames[0].shape[:2]
                io_adapter = IOAdapter(
                    output_stride=model.output_stride,
                    input_size=frame_shape,
                    cuda=torch.cuda.is_available(),
                )

            # Compute GT flows
            flows = []
            for i in tqdm(range(len(frames) - 1), desc="    Flow", leave=False):
                flow = compute_flow_pair(model, io_adapter, frames[i], frames[i + 1])
                flows.append(flow)

            # Load actions
            actions = np.load(str(action_path), allow_pickle=True).item()

            # Build regression data
            A, b = build_regression_data(
                flows, actions, focal_length,
                n_pixels_per_frame=n_pixels_per_frame,
                rng=rng,
            )
            all_A.append(A)
            all_b.append(b)

    if not all_A:
        raise RuntimeError("No data collected for calibration!")

    # Stack all data
    A_full = np.vstack(all_A)
    b_full = np.concatenate(all_b)

    print(f"\nRegression: {A_full.shape[0]} equations, {A_full.shape[1]} unknowns")

    # Filter out rows where all design columns are zero (no action → uninformative)
    row_nonzero = np.any(np.abs(A_full) > 1e-10, axis=1)
    n_nonzero = row_nonzero.sum()
    print(f"  Non-zero rows: {n_nonzero}/{len(row_nonzero)}")

    if n_nonzero < 10:
        raise RuntimeError("Too few informative data points for calibration!")

    A_reg = A_full[row_nonzero]
    b_reg = b_full[row_nonzero]

    # Solve with least squares
    result, residuals, rank, sv = np.linalg.lstsq(A_reg, b_reg, rcond=None)

    alpha_yaw, alpha_pitch, alpha_turn, beta_fwd, beta_strafe = result

    # Compute fit quality
    b_pred = A_reg @ result
    residual_rms = float(np.sqrt(np.mean((b_reg - b_pred) ** 2)))
    print(f"  Residual RMS: {residual_rms:.4f} px")

    calibration = {
        "alpha_yaw": float(alpha_yaw),
        "alpha_pitch": float(alpha_pitch),
        "alpha_turn": float(alpha_turn),
        "beta_fwd": float(beta_fwd),
        "beta_strafe": float(beta_strafe),
        "focal_length": focal_length,
        "frame_shape": list(frame_shape) if frame_shape is not None else None,
        "calibrated_from": scenarios,
        "residual_rms": residual_rms,
        "n_equations": int(n_nonzero),
    }

    print(f"\nCalibration results:")
    print(f"  alpha_yaw:   {alpha_yaw:.6f}")
    print(f"  alpha_pitch: {alpha_pitch:.6f}")
    print(f"  alpha_turn:  {alpha_turn:.6f}")
    print(f"  beta_fwd:    {beta_fwd:.6f}")
    print(f"  beta_strafe: {beta_strafe:.6f}")
    print(f"  focal_length: {focal_length}")

    return calibration


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate synthetic flow parameters from GT videos"
    )
    parser.add_argument("--assets_dir", type=str, default="assets",
                        help="Root assets directory")
    parser.add_argument("--output", type=str, default="calibration.json",
                        help="Output calibration JSON file")
    parser.add_argument("--scenarios", type=str, nargs="*", default=None,
                        help="Scenarios to use (default: all)")
    parser.add_argument("--model", type=str, default="dpflow",
                        help="Optical flow model")
    parser.add_argument("--ckpt", type=str, default="things",
                        help="Model checkpoint")
    parser.add_argument("--focal_length", type=float, default=457.0,
                        help="Focal length in pixels (default: 457 for 70deg FOV)")
    parser.add_argument("--n_pixels", type=int, default=500,
                        help="Pixels to sample per frame for regression")
    args = parser.parse_args()

    # Discover scenarios if not specified
    if args.scenarios is None:
        assets_root = Path(args.assets_dir)
        args.scenarios = sorted(
            d.name for d in assets_root.iterdir() if d.is_dir()
        )

    print(f"Calibrating from scenarios: {args.scenarios}")
    t0 = time.time()

    calibration = calibrate(
        assets_dir=args.assets_dir,
        scenarios=args.scenarios,
        model_name=args.model,
        ckpt=args.ckpt,
        focal_length=args.focal_length,
        n_pixels_per_frame=args.n_pixels,
    )

    # Save
    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nCalibration saved to {args.output}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
