from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_significant_tokens(significant_tokens_file: Optional[str]) -> List[str]:
    if significant_tokens_file is None:
        return []

    path = Path(significant_tokens_file)
    if not path.exists():
        return []

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def save_visualization(
    tokens: List[str],
    zip_scores: List[float],
    significant_tokens: List[str],
    out_png: str,
):

    if len(tokens) == 0:
        raise ValueError("No tokens to visualize.")

    scores = zip_scores[: len(tokens)]

    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-12:
        normalized = [0.0 for _ in scores]
    else:
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]

    non_sig = [t for t in tokens if t not in significant_tokens]

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(tokens)), 2))
    cmap = plt.get_cmap("Greens")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    temp_texts = []
    for word, score in zip(tokens, normalized):
        background_color = cmap(score)
        text_color = "black" if score < 0.5 else "white"
        edge_color = "brown" if word not in non_sig else "none"
        linewidth = 3 if word not in non_sig else 0

        txt = ax.text(
            0.0,
            0.5,
            word,
            ha="left",
            va="center",
            color=text_color,
            transform=ax.transAxes,
            alpha=0.0,
            bbox=dict(
                facecolor=background_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                boxstyle="round,pad=0.6",
            ),
        )
        temp_texts.append(txt)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)

    box_widths_px = [txt.get_window_extent(renderer=renderer).width for txt in temp_texts]

    for txt in temp_texts:
        txt.remove()

    gap_px = 24  # change this if you want tighter/looser spacing

    total_boxes_width_px = sum(box_widths_px)
    total_gap_width_px = gap_px * (len(tokens) - 1)
    total_content_width_px = total_boxes_width_px + total_gap_width_px

    # Center the whole row horizontally in the axes
    start_left_px = axes_bbox.x0 + max(0, (axes_bbox.width - total_content_width_px) / 2)

    positions = []
    current_left_px = start_left_px

    for width_px in box_widths_px:
        center_px = current_left_px + width_px / 2
        center_axes = (center_px - axes_bbox.x0) / axes_bbox.width
        positions.append(center_axes)

        current_left_px += width_px + gap_px

    for i, (word, score) in enumerate(zip(tokens, normalized)):
        background_color = cmap(score)
        text_color = "black" if score < 0.5 else "white"
        edge_color = "brown" if word not in non_sig else "none"
        linewidth = 3 if word not in non_sig else 0

        ax.text(
            positions[i],
            0.5,
            word,
            ha="center",
            va="center",
            color=text_color,
            transform=ax.transAxes,
            bbox=dict(
                facecolor=background_color,
                edgecolor=edge_color,
                linewidth=linewidth,
                boxstyle="round,pad=0.6",
            ),
        )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)

    return fig


def plot_zip_heatmap(
    prompt: str,
    zip_scores_file: str,
    output_path: Optional[str] = None,
    significant_tokens_file: Optional[str] = None,
    show: bool = True,
):

    scores_df = pd.read_csv(zip_scores_file)
    significant_tokens = load_significant_tokens(significant_tokens_file)

    tokens = prompt.split()

    score_map = {
        int(row["token_index"]): float(row["zip_score"])
        for _, row in scores_df.iterrows()
    }

    zip_scores = [score_map.get(i, 0.0) for i in range(len(tokens))]

    if output_path is None:
        output_path = "zip_heatmap.png"

    fig = save_visualization(
        tokens=tokens,
        zip_scores=zip_scores,
        significant_tokens=significant_tokens,
        out_png=output_path,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig