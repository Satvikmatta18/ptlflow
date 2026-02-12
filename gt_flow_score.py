"""Score video generation quality by comparing optical flow against a ground truth video.

Computes optical flow from both the GT and generated videos, then returns
pixel_epe_mean: the average end-point error between GT and generated flow
across all frames.

Lower score = better visual fidelity to ground truth motion.

Usage:
    # As a library
    from gt_flow_score import GTFlowScorer
    scorer = GTFlowScorer()
    score = scorer.score("gt.mp4", "gen.mp4")

    # From command line
    python gt_flow_score.py --gt_video gt.mp4 --gen_video gen.mp4
"""

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
    flow_gt: np.ndarray,
    flow_gen: np.ndarray,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> float:
    """Compute mean end-point error between two flow fields.

    Filters out near-zero vectors (< min_mag) and extreme outliers
    (top percentile by magnitude) to reduce noise.
    """
    gt_mag = np.linalg.norm(flow_gt, axis=2)
    gen_mag = np.linalg.norm(flow_gen, axis=2)
    max_mag = np.maximum(gt_mag, gen_mag)

    mag_hi = np.percentile(max_mag, max_mag_pct)
    mask = (max_mag >= min_mag) & (max_mag <= mag_hi)

    epe_map = np.linalg.norm(flow_gt - flow_gen, axis=2)
    if mask.sum() > 0:
        return float(epe_map[mask].mean())
    return float(epe_map.mean())


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class GTFlowScorer:
    """Score generated videos by comparing optical flow against ground truth.

    Loads the optical flow model once, then scores video pairs efficiently
    via the `score()` method.

    Parameters
    ----------
    model_name : str
        Optical flow model name (default: "dpflow").
    ckpt : str
        Model checkpoint (default: "things").
    """

    def __init__(
        self,
        model_name: str = "dpflow",
        ckpt: str = "things",
    ):
        self.model = ptlflow.get_model(model_name, ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self._io_adapter = None

    def _get_io_adapter(self, frame_shape: Tuple[int, int]) -> IOAdapter:
        if self._io_adapter is None:
            self._io_adapter = IOAdapter(
                output_stride=self.model.output_stride,
                input_size=frame_shape,
                cuda=torch.cuda.is_available(),
            )
        return self._io_adapter

    @torch.no_grad()
    def score(
        self,
        gt_video: Union[str, Path],
        gen_video: Union[str, Path],
    ) -> float:
        """Score a generated video against its ground truth.

        Parameters
        ----------
        gt_video : str or Path
            Path to the ground truth video (.mp4).
        gen_video : str or Path
            Path to the generated video (.mp4).

        Returns
        -------
        float
            pixel_epe_mean — average end-point error across all frames.
            Lower is better.
        """
        frames_gt = extract_frames(str(gt_video))
        frames_gen = extract_frames(str(gen_video))

        if len(frames_gt) < 2:
            raise ValueError(f"GT video has {len(frames_gt)} frames, need at least 2")
        if len(frames_gen) < 2:
            raise ValueError(f"Gen video has {len(frames_gen)} frames, need at least 2")

        # Align lengths
        n_use = min(len(frames_gt), len(frames_gen))
        frames_gt = frames_gt[:n_use]
        frames_gen = frames_gen[:n_use]

        # Get IO adapter
        shape = frames_gt[0].shape[:2]
        io_adapter = self._get_io_adapter(shape)

        # Compute flow for both videos
        flows_gt = compute_flow_sequence(self.model, frames_gt, io_adapter)
        flows_gen = compute_flow_sequence(self.model, frames_gen, io_adapter)

        # Compute mean EPE across all frames
        n_flows = len(flows_gt)
        epe_sum = 0.0
        for i in range(n_flows):
            epe_sum += compute_pixel_epe(flows_gt[i], flows_gen[i])

        return epe_sum / n_flows

    def score_batch(
        self,
        video_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
    ) -> List[float]:
        """Score multiple video pairs. Returns list of pixel_epe_mean scores."""
        return [self.score(gt, gen) for gt, gen in video_pairs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Score video generation quality against ground truth."
    )
    parser.add_argument("--gt_video", type=str, required=True,
                        help="Path to ground truth video (.mp4)")
    parser.add_argument("--gen_video", type=str, required=True,
                        help="Path to generated video (.mp4)")
    parser.add_argument("--model", type=str, default="dpflow",
                        help="Optical flow model (default: dpflow)")
    parser.add_argument("--ckpt", type=str, default="things",
                        help="Model checkpoint (default: things)")
    args = parser.parse_args()

    scorer = GTFlowScorer(
        model_name=args.model,
        ckpt=args.ckpt,
    )
    score = scorer.score(args.gt_video, args.gen_video)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
