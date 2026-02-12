"""Visualize flow evaluation results alongside GT/Gen videos and action overlays.

Reads .npy eval files from flow_eval_all/ and pairs them with assets/ videos + actions.
Produces side-by-side comparison videos with action overlays and time-series metric plots.

Output: one .mp4 per video pair in output_dir/{scenario}/{video_id}.mp4

Usage:
    python visualize_results.py --assets_dir assets --eval_dir outputs/flow_eval_all
    python visualize_results.py --assets_dir /local/path/assets --eval_dir /local/path/flow_eval_all
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Action drawing helpers (adapted from FastVideo wangame_training_pipeline.py)
# ---------------------------------------------------------------------------

KEY_NAMES = ["W", "S", "A", "D", "left", "right"]
KEY_ICONS = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "L", "right": "R"}


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_keys_on_frame(frame, keys_pressed, key_size=(30, 30), top_margin=15):
    """Draw WASD + L/R key indicators on top-left of frame."""
    left_margin = 15
    gap = 3
    key_positions = {
        "W": (left_margin + key_size[0] + gap, top_margin),
        "A": (left_margin, top_margin + key_size[1] + gap),
        "S": (left_margin + key_size[0] + gap, top_margin + key_size[1] + gap),
        "D": (left_margin + (key_size[0] + gap) * 2, top_margin + key_size[1] + gap),
        "left": (left_margin + (key_size[0] + gap) * 3 + 10, top_margin + key_size[1] + gap),
        "right": (left_margin + (key_size[0] + gap) * 4 + 15, top_margin + key_size[1] + gap),
    }

    for key, (x, y) in key_positions.items():
        is_pressed = keys_pressed.get(key, False)
        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5
        draw_rounded_rectangle(frame, (x, y), (x + key_size[0], y + key_size[1]),
                               color, radius=5, alpha=alpha)
        icon = KEY_ICONS[key]
        text_size = cv2.getTextSize(icon, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, icon, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_mouse_on_frame(frame, pitch, yaw, top_margin=15):
    """Draw crosshair with direction arrow on top-right of frame."""
    h, w = frame.shape[:2]
    right_margin = 15
    r = 25
    cx = w - right_margin - r
    cy = top_margin + r

    dx = int(yaw * r * 8)
    dy = int(-pitch * r * 8)
    max_arrow = r - 5
    dx = max(-max_arrow, min(max_arrow, dx))
    dy = max(-max_arrow, min(max_arrow, dy))

    cv2.circle(frame, (cx, cy), r, (50, 50, 50), -1)
    cv2.circle(frame, (cx, cy), r, (200, 200, 200), 1)
    cv2.line(frame, (cx - r + 5, cy), (cx + r - 5, cy), (100, 100, 100), 1)
    cv2.line(frame, (cx, cy - r + 5), (cx, cy + r - 5), (100, 100, 100), 1)
    if abs(dx) > 1 or abs(dy) > 1:
        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                        (0, 255, 0), 2, tipLength=0.3)


def overlay_actions(frame, keyboard_vec, mouse_vec):
    """Overlay keyboard and mouse indicators on a frame (in-place)."""
    keys_pressed = {name: bool(keyboard_vec[i] > 0.5)
                    for i, name in enumerate(KEY_NAMES)}
    draw_keys_on_frame(frame, keys_pressed)
    if mouse_vec is not None and len(mouse_vec) >= 2:
        draw_mouse_on_frame(frame, float(mouse_vec[0]), float(mouse_vec[1]))


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps if fps > 0 else 10.0


# ---------------------------------------------------------------------------
# Time-series plot rendering
# ---------------------------------------------------------------------------

METRIC_GROUPS = [
    {
        "title": "Pixel EPE (mean)",
        "metrics": [("pixel_epe_mean", "EPE", "#e74c3c")],
    },
    {
        "title": "Direction",
        "metrics": [("mf_cosine", "Cosine Sim", "#2ecc71")],
        "metrics_right": [("mf_angle_err", "Angle Err (deg)", "#e67e22")],
    },
    {
        "title": "Flow Outlier (Fl-all)",
        "metrics": [("fl_all", "Fl-all", "#9b59b6")],
    },
    {
        "title": "FOE Distance",
        "metrics": [("foe_dist", "FOE dist (px)", "#3498db")],
    },
    {
        "title": "KL Divergence (2D)",
        "metrics": [("flow_kl_2d", "KL div", "#e74c3c")],
    },
]


def render_timeseries(frame_metrics, current_frame, width, height, dpi=100):
    """Render the time-series plot as a numpy BGR image."""
    n_groups = len(METRIC_GROUPS)
    fig_w = width / dpi
    fig_h = height / dpi
    fig, axes = plt.subplots(n_groups, 1, figsize=(fig_w, fig_h), dpi=dpi,
                             sharex=True)
    if n_groups == 1:
        axes = [axes]

    n_frames = len(frame_metrics)
    x = list(range(n_frames))

    for ax, group in zip(axes, METRIC_GROUPS):
        # Left axis
        for metric_key, label, color in group["metrics"]:
            vals = [m.get(metric_key, 0) for m in frame_metrics]
            ax.plot(x, vals, color=color, linewidth=1.2, label=label)
        ax.set_ylabel(group["metrics"][0][1], fontsize=7, color=group["metrics"][0][2])
        ax.tick_params(axis="y", labelsize=6, colors=group["metrics"][0][2])
        ax.set_title(group["title"], fontsize=8, pad=2)
        ax.grid(True, alpha=0.3)

        # Right axis if present
        if "metrics_right" in group:
            ax2 = ax.twinx()
            for metric_key, label, color in group["metrics_right"]:
                vals = [m.get(metric_key, 0) for m in frame_metrics]
                ax2.plot(x, vals, color=color, linewidth=1.2, linestyle="--", label=label)
            ax2.set_ylabel(group["metrics_right"][0][1], fontsize=7,
                           color=group["metrics_right"][0][2])
            ax2.tick_params(axis="y", labelsize=6, colors=group["metrics_right"][0][2])

        # Cursor line
        ax.axvline(current_frame, color="lime", linewidth=1.5, alpha=0.8)

    axes[-1].set_xlabel("Frame", fontsize=8)
    axes[-1].tick_params(axis="x", labelsize=6)
    plt.tight_layout(pad=0.5)

    # Render to numpy
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h_px, w_px, 4)[:, :, :3]  # drop alpha
    plt.close(fig)
    # RGB -> BGR for OpenCV
    return buf[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Label overlay
# ---------------------------------------------------------------------------

def put_label(frame, text, position="top-center"):
    """Put a semi-transparent label on a frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    tw, th = text_size

    if position == "top-center":
        tx = (w - tw) // 2
        ty = th + 8
    elif position == "top-left":
        tx = 10
        ty = th + 8
    else:
        tx = (w - tw) // 2
        ty = th + 8

    # Background rectangle
    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pair(eval_npy_path, assets_dir, output_path, target_width=1280):
    """Process a single GT/Gen video pair and produce the visualization video.

    Supports both GT mode and synthetic mode. In synthetic mode, only the
    generated video is shown (no GT video needed).
    """
    # Load eval data
    data = np.load(str(eval_npy_path), allow_pickle=True).item()
    frame_metrics = data["frame_metrics"]
    summary = data["summary"]
    mode = data.get("mode", "gt")

    # Determine scenario/video_id from path
    scenario = eval_npy_path.parent.name
    video_id = eval_npy_path.stem

    # Load videos
    gen_path = assets_dir / scenario / f"{video_id}_wangame.mp4"
    action_path = assets_dir / scenario / f"{video_id}_action.npy"

    if not gen_path.exists():
        print(f"  Skipping: missing generated video for {scenario}/{video_id}")
        return False

    frames_gen, fps = extract_frames(gen_path)

    # Load GT video if in GT mode
    frames_gt = None
    if mode != "synthetic":
        gt_path = assets_dir / scenario / f"{video_id}.mp4"
        if gt_path.exists():
            frames_gt, fps = extract_frames(gt_path)
        else:
            print(f"  Warning: GT video not found, showing gen-only layout")

    # Load actions
    actions = None
    if action_path.exists():
        actions = np.load(str(action_path), allow_pickle=True).item()

    if frames_gt is not None:
        n_frames = min(len(frames_gt), len(frames_gen))
        frames_gt = frames_gt[:n_frames]
        frames_gen = frames_gen[:n_frames]
    else:
        n_frames = len(frames_gen)

    if n_frames < 2:
        print(f"  Skipping: need at least 2 frames")
        return False

    # Frame dimensions
    fh, fw = frames_gen[0].shape[:2]
    panel_w = target_width // 2
    scale = panel_w / fw
    panel_h = int(fh * scale)

    plot_w = target_width
    plot_h = int(panel_h * 1.2)

    canvas_w = target_width
    canvas_h = panel_h + plot_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    n_flow_frames = len(frame_metrics)

    # Labels based on mode
    if mode == "synthetic":
        left_label = "Generated (WanGame)"
        right_label = "Synthetic Flow Reference"
    else:
        left_label = "Ground Truth"
        right_label = "Generated (WanGame)"

    for i in range(n_frames):
        gen_frame = frames_gen[i].copy()

        # Overlay actions on gen frame
        if actions is not None:
            keyboard = actions["keyboard"]
            mouse = actions["mouse"]
            action_idx = min(i, len(keyboard) - 1)
            overlay_actions(gen_frame, keyboard[action_idx], mouse[action_idx])

        gen_resized = cv2.resize(gen_frame, (panel_w, panel_h))

        if frames_gt is not None and mode != "synthetic":
            gt_frame = frames_gt[i].copy()
            if actions is not None:
                overlay_actions(gt_frame, keyboard[action_idx], mouse[action_idx])
            left_panel = cv2.resize(gt_frame, (panel_w, panel_h))
            put_label(left_panel, left_label)
            # Frame counter on GT
            cv2.putText(left_panel, f"Frame {i}/{n_frames - 1}",
                        (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            right_panel = gen_resized
            put_label(right_panel, right_label)
        else:
            # Synthetic mode or missing GT: gen on left, info on right
            left_panel = gen_resized
            put_label(left_panel, left_label)
            cv2.putText(left_panel, f"Frame {i}/{n_frames - 1}",
                        (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            # Info panel on right
            right_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            put_label(right_panel, "Synthetic Flow Mode")
            cv2.putText(right_panel, "Ref: action-based flow",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)

        # Metric overlay on right panel
        if i < n_flow_frames:
            m = frame_metrics[i]
            lines = [
                f"EPE: {m.get('pixel_epe_mean', 0):.2f}",
                f"Fl-all: {m.get('fl_all', 0) * 100:.1f}%",
                f"Cosine: {m.get('mf_cosine', 0):.4f}",
                f"Angle: {m.get('mf_angle_err', 0):.1f} deg",
                f"KL: {m.get('flow_kl_2d', 0):.3f}",
            ]
            target = right_panel if mode != "synthetic" else left_panel
            y0 = panel_h - 10 - (len(lines) - 1) * 18
            for j, line in enumerate(lines):
                cv2.putText(target, line,
                            (10, y0 + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)

        video_row = np.hstack([left_panel, right_panel])

        cursor = min(i, n_flow_frames - 1)
        plot_img = render_timeseries(frame_metrics, cursor, plot_w, plot_h)
        plot_img = cv2.resize(plot_img, (plot_w, plot_h))

        canvas = np.vstack([video_row, plot_img])
        writer.write(canvas)

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize flow eval results with GT/Gen videos and actions")
    parser.add_argument("--assets_dir", type=str, required=True,
                        help="Path to assets/ directory with videos and actions")
    parser.add_argument("--eval_dir", type=str, default="outputs/flow_eval_all",
                        help="Path to flow_eval_all/ directory with .npy results")
    parser.add_argument("--output_dir", type=str, default="outputs/viz_results",
                        help="Output directory for visualization videos")
    parser.add_argument("--width", type=int, default=1280,
                        help="Target video width (default: 1280)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Only process this scenario (e.g. 'camera')")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Only process this video ID (e.g. '01')")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)

    if not assets_dir.exists():
        print(f"Error: assets directory not found: {assets_dir}")
        return
    if not eval_dir.exists():
        print(f"Error: eval directory not found: {eval_dir}")
        return

    # Discover all eval .npy files
    npy_files = sorted(eval_dir.glob("*/*.npy"))
    if args.scenario:
        npy_files = [f for f in npy_files if f.parent.name == args.scenario]
    if args.video_id:
        npy_files = [f for f in npy_files if f.stem == args.video_id]

    print(f"Found {len(npy_files)} eval files to visualize")

    for idx, npy_path in enumerate(npy_files):
        scenario = npy_path.parent.name
        video_id = npy_path.stem
        out_path = output_dir / scenario / f"{video_id}.mp4"

        print(f"[{idx + 1}/{len(npy_files)}] {scenario}/{video_id} -> {out_path}")
        ok = process_pair(npy_path, assets_dir, out_path, target_width=args.width)
        if ok:
            print(f"  Done.")
        else:
            print(f"  Failed.")

    print(f"\nVisualization videos saved to: {output_dir}/")


if __name__ == "__main__":
    main()
