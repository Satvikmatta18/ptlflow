"""Score action-conditioned video generation quality using synthetic optical flow.

Compares synthetic flow (derived analytically from game actions) against actual
optical flow extracted from the generated video. Returns pixel_epe_mean: the
average end-point error between expected and actual flow across all frames.

Lower score = better action adherence.

No depth estimation is used — translation flow assumes constant depth (Z=1).
This is simpler, faster, and empirically more correlated with GT-based metrics.

Usage:
    # As a library
    from action_flow_score import ActionFlowScorer
    scorer = ActionFlowScorer("calibration.json")
    score = scorer.score("video.mp4", "action.npy")

    # From command line
    python action_flow_score.py --video video.mp4 --actions action.npy
    python action_flow_score.py --video video.mp4 --actions action.npy --calibration calibration.json
"""

import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import tensor_dict_to_numpy


# ---------------------------------------------------------------------------
# Synthetic flow generation (no depth, Longuet-Higgins model)
# ---------------------------------------------------------------------------

class SyntheticFlowGenerator:
    """Generate optical flow from game actions using the Longuet-Higgins model.

    Rotation flow is depth-independent (quadratic spatial terms).
    Translation flow uses constant depth Z=1 (no depth model needed).
    """

    def __init__(self, calibration: dict, frame_shape: Tuple[int, int]):
        self.alpha_yaw = calibration["alpha_yaw"]
        self.alpha_pitch = calibration["alpha_pitch"]
        self.alpha_turn = calibration["alpha_turn"]
        self.beta_fwd = calibration["beta_fwd"]
        self.beta_strafe = calibration["beta_strafe"]
        self.f = calibration["focal_length"]

        H, W = frame_shape
        cx, cy = W / 2.0, H / 2.0
        xs = np.arange(W, dtype=np.float64) - cx
        ys = np.arange(H, dtype=np.float64) - cy
        self.x_grid, self.y_grid = np.meshgrid(xs, ys)

        f = self.f
        self.xy_over_f = self.x_grid * self.y_grid / f
        self.f_plus_x2_over_f = f + self.x_grid ** 2 / f
        self.f_plus_y2_over_f = f + self.y_grid ** 2 / f

    def generate_flow(self, keyboard: np.ndarray, mouse: np.ndarray) -> np.ndarray:
        """Generate HxWx2 synthetic flow for one frame's actions.

        Parameters
        ----------
        keyboard : shape (6,) — [W, S, A, D, left, right]
        mouse : shape (2,) — [pitch, yaw]
        """
        # Rotation velocities
        omega_y = self.alpha_yaw * mouse[1] + self.alpha_turn * (keyboard[5] - keyboard[4])
        omega_x = self.alpha_pitch * mouse[0]

        # Rotation flow (depth-independent)
        u = (self.xy_over_f * omega_x
             - self.f_plus_x2_over_f * omega_y)
        v = (self.f_plus_y2_over_f * omega_x
             - self.xy_over_f * omega_y)
        flow = np.stack([u, v], axis=-1)

        # Translation velocities
        Tz = self.beta_fwd * (keyboard[0] - keyboard[1])
        Tx = self.beta_strafe * (keyboard[3] - keyboard[2])

        if abs(Tx) > 1e-8 or abs(Tz) > 1e-8:
            # Translation flow with constant depth Z=1
            u_t = -self.f * Tx + self.x_grid * Tz
            v_t = self.y_grid * Tz  # Ty = 0
            flow = flow + np.stack([u_t, v_t], axis=-1)

        return flow.astype(np.float32)


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str) -> List[np.ndarray]:
    """Read all frames from a video file as BGR uint8 arrays."""
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


# ---------------------------------------------------------------------------
# Optical flow from video
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_flow_sequence(
    model: torch.nn.Module,
    frames: List[np.ndarray],
    io_adapter: IOAdapter,
) -> List[np.ndarray]:
    """Compute optical flow for consecutive frame pairs. Returns list of HxWx2."""
    flows = []
    for i in tqdm(range(len(frames) - 1), desc="Computing flow", leave=False):
        inputs = io_adapter.prepare_inputs([frames[i], frames[i + 1]])
        preds = model(inputs)
        preds["images"] = inputs["images"]
        preds = io_adapter.unscale(preds)
        preds_npy = tensor_dict_to_numpy(preds)
        flows.append(preds_npy["flows"])
    return flows


# ---------------------------------------------------------------------------
# Metric: pixel EPE
# ---------------------------------------------------------------------------

def compute_pixel_epe(
    flow_synth: np.ndarray,
    flow_gen: np.ndarray,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> float:
    """Compute mean end-point error between two flow fields.

    Filters out near-zero vectors (< min_mag) and extreme outliers
    (top percentile by magnitude) to reduce noise.
    """
    synth_mag = np.linalg.norm(flow_synth, axis=2)
    gen_mag = np.linalg.norm(flow_gen, axis=2)
    max_mag = np.maximum(synth_mag, gen_mag)

    mag_hi = np.percentile(max_mag, max_mag_pct)
    mask = (max_mag >= min_mag) & (max_mag <= mag_hi)

    epe_map = np.linalg.norm(flow_synth - flow_gen, axis=2)
    if mask.sum() > 0:
        return float(epe_map[mask].mean())
    return float(epe_map.mean())


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class ActionFlowScorer:
    """Score action-conditioned videos by comparing synthetic vs actual flow.

    Loads the optical flow model and calibration once, then scores videos
    efficiently via the `score()` method.

    Parameters
    ----------
    calibration : str or dict
        Path to calibration.json, or a dict with calibration coefficients.
    model_name : str
        Optical flow model name (default: "dpflow").
    ckpt : str
        Model checkpoint (default: "things").
    """

    def __init__(
        self,
        calibration: Union[str, dict] = "calibration.json",
        model_name: str = "dpflow",
        ckpt: str = "things",
    ):
        # Load calibration
        if isinstance(calibration, (str, Path)):
            with open(calibration) as f:
                self.calibration = json.load(f)
        else:
            self.calibration = calibration

        # Load flow model
        self.model = ptlflow.get_model(model_name, ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self._io_adapter = None
        self._flow_generator = None

    def _get_io_adapter(self, frame_shape: Tuple[int, int]) -> IOAdapter:
        if self._io_adapter is None:
            self._io_adapter = IOAdapter(
                output_stride=self.model.output_stride,
                input_size=frame_shape,
                cuda=torch.cuda.is_available(),
            )
        return self._io_adapter

    def _get_flow_generator(self, frame_shape: Tuple[int, int]) -> SyntheticFlowGenerator:
        if self._flow_generator is None:
            self._flow_generator = SyntheticFlowGenerator(
                calibration=self.calibration,
                frame_shape=frame_shape,
            )
        return self._flow_generator

    @torch.no_grad()
    def score(
        self,
        video_path: Union[str, Path],
        action_path: Union[str, Path],
    ) -> float:
        """Score a single video against its actions.

        Parameters
        ----------
        video_path : str or Path
            Path to the generated video (.mp4).
        action_path : str or Path
            Path to the action file (.npy) with keys "keyboard" (T,6) and "mouse" (T,2).

        Returns
        -------
        float
            pixel_epe_mean — average end-point error across all frames.
            Lower is better.
        """
        # Extract frames
        frames = extract_frames(str(video_path))
        n_frames = len(frames)
        if n_frames < 2:
            raise ValueError(f"Video has {n_frames} frames, need at least 2")

        # Load actions
        actions = np.load(str(action_path), allow_pickle=True).item()
        keyboard = actions["keyboard"]
        mouse = actions["mouse"]
        n_actions = len(keyboard)

        # Align lengths
        n_use = min(n_frames, n_actions)
        frames = frames[:n_use]

        # Get adapters
        shape = frames[0].shape[:2]
        io_adapter = self._get_io_adapter(shape)
        flow_gen = self._get_flow_generator(shape)

        # Compute actual flow from video
        flows_actual = compute_flow_sequence(self.model, frames, io_adapter)

        # Generate synthetic flow from actions (T-1 flows)
        n_flows = len(flows_actual)
        epe_sum = 0.0
        for i in range(n_flows):
            flow_synth = flow_gen.generate_flow(keyboard[i], mouse[i])
            epe_sum += compute_pixel_epe(flow_synth, flows_actual[i])

        return epe_sum / n_flows

    def score_batch(
        self,
        video_action_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
    ) -> List[float]:
        """Score multiple videos. Returns list of pixel_epe_mean scores."""
        return [self.score(v, a) for v, a in video_action_pairs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Score action-conditioned video generation quality."
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to generated video (.mp4)")
    parser.add_argument("--actions", type=str, required=True,
                        help="Path to action file (.npy)")
    parser.add_argument("--calibration", type=str, default="calibration.json",
                        help="Path to calibration.json (default: calibration.json)")
    parser.add_argument("--model", type=str, default="dpflow",
                        help="Optical flow model (default: dpflow)")
    parser.add_argument("--ckpt", type=str, default="things",
                        help="Model checkpoint (default: things)")
    args = parser.parse_args()

    scorer = ActionFlowScorer(
        calibration=args.calibration,
        model_name=args.model,
        ckpt=args.ckpt,
    )
    score = scorer.score(args.video, args.actions)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
