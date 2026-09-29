"""
Publication figures: landscape heatmap, AR bar chart, ZNE curve, results table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_landscape(
    grids: list[np.ndarray],
    labels: list[str],
    gammas: np.ndarray,
    betas: np.ndarray,
    gamma_star: float,
    beta_star: float,
    best_params: list[tuple],
    save_path: str | Path | None = None,
    show_diff: bool = True,
) -> plt.Figure:
    """
    Figure 1: side-by-side E(γ, β) heatmaps.

    When show_diff=True and exactly 2 grids are provided, appends a third
    panel showing (grids[1] - grids[0]) with a diverging colormap. Each
    energy panel gets its own colorbar so the full dynamic range is visible.

    Star marker at (γ*, β*) — classically optimal point.
    Cross marker at each condition's own grid minimum.

    Args:
        grids:       List of (n_points × n_points) energy arrays.
        labels:      Panel titles (one per grid).
        gammas:      1-D array of γ grid values (y-axis).
        betas:       1-D array of β grid values (x-axis).
        gamma_star:  γ* from COBYLA optimisation.
        beta_star:   β* from COBYLA optimisation.
        best_params: List of (gamma, beta) best grid points, one per grid.
        save_path:   If given, saves as both .pdf and .png (300 DPI).
        show_diff:   If True and len(grids)==2, add a (noisy−ideal) difference panel.

    Returns:
        matplotlib Figure.
    """
    add_diff = show_diff and len(grids) == 2
    n_panels = len(grids) + (1 if add_diff else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5), sharey=True,
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # Energy panels — per-panel colorbar so each landscape uses its full range
    for ax, grid, label, bp in zip(axes[:len(grids)], grids, labels, best_params):
        im = ax.pcolormesh(betas, gammas, grid, cmap='RdYlBu_r', shading='nearest')
        fig.colorbar(im, ax=ax, label='E(γ, β)', fraction=0.046, pad=0.04)
        ax.plot(beta_star, gamma_star, '*', color='white', markersize=14,
                markeredgecolor='black', markeredgewidth=0.8, label='(γ*, β*)')
        ax.plot(bp[1], bp[0], 'x', color='lime', markersize=10,
                markeredgewidth=2.5, label='Grid best')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('β', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')

    axes[0].set_ylabel('γ', fontsize=12)

    # Difference panel
    if add_diff:
        diff = grids[1] - grids[0]
        abs_max = np.abs(diff).max()
        ax_diff = axes[-1]
        im_diff = ax_diff.pcolormesh(betas, gammas, diff, cmap='RdBu_r',
                                     vmin=-abs_max, vmax=abs_max, shading='nearest')
        fig.colorbar(im_diff, ax=ax_diff, label='ΔE (noisy − ideal)', fraction=0.046, pad=0.04)
        ax_diff.plot(beta_star, gamma_star, '*', color='black', markersize=14,
                     markeredgecolor='white', markeredgewidth=0.8)
        ax_diff.set_title(f'Difference\n(mean |ΔE|={np.abs(diff).mean():.2f}, max={abs_max:.2f})',
                          fontsize=13)
        ax_diff.set_xlabel('β', fontsize=12)

    fig.suptitle('QAOA Energy Landscape (p=1)', fontsize=14)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    return fig


def plot_approximation_ratios(
    ar_values: dict[str, float],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Figure 2: approximation ratio bar chart across conditions.

    Args:
        ar_values: Dict mapping condition label → AR value.
        save_path: Optional path prefix (.pdf and .png saved).

    Returns:
        matplotlib Figure.
    """
    labels = list(ar_values.keys())
    values = list(ar_values.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=['steelblue', 'darkorange', 'firebrick', 'seagreen'])
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', label='Optimal (AR=1)')
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Approximation Ratio by Condition', fontsize=13)
    ax.set_ylim(min(values) - 0.05, 1.05)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    return fig


def plot_zne_curve(
    scale_factors: list[int],
    energies: list[float],
    e_zne: float,
    e_raw: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Figure 3: ZNE extrapolation curve.

    Args:
        scale_factors: Noise scale factors used (e.g. [1, 3, 5]).
        energies:      Measured energies at each scale factor.
        e_zne:         Extrapolated ZNE energy (at scale=0).
        e_raw:         Raw hardware energy (scale=1) for reference.
        save_path:     Optional path prefix.

    Returns:
        matplotlib Figure.
    """
    coeffs = np.polyfit(scale_factors, energies, deg=1)
    x_fit  = np.linspace(0, max(scale_factors), 200)
    y_fit  = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_fit, y_fit, '--', color='steelblue', label='Linear fit')
    ax.scatter(scale_factors, energies, color='steelblue', zorder=5, label='Measured')
    ax.scatter([0], [e_zne], marker='*', color='red', s=150, zorder=6, label=f'ZNE = {e_zne:.4f}')
    ax.axhline(e_raw, color='grey', linewidth=0.8, linestyle=':', label=f'Raw (scale=1) = {e_raw:.4f}')

    ax.set_xlabel('Noise scale factor', fontsize=12)
    ax.set_ylabel('E(γ*, β*)', fontsize=12)
    ax.set_title('Zero Noise Extrapolation', fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    return fig
