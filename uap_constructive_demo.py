from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


def paper_like_polynomials(count: int) -> list[Polynomial]:
    """Return a short monic polynomial sequence inspired by the paper's dense family.

    The paper uses a dense sequence of monic polynomials. For a runnable demo,
    we use a finite prefix with the same spirit: simple monic polynomials with
    mixed lower-order terms.
    """

    x = Polynomial([0.0, 1.0])
    candidates = [
        Polynomial([1.0]),
        x,
        x**2,
        x**2 - x,
        x**2 - 1.0,
        x**3,
        x**3 - x,
        x**3 - 1.0,
        x**4,
        x**4 - x,
        x**4 - 1.0,
        x**4 + x,
        x**4 + x - 1.0,
        x**5,
        x**5 - x,
    ]
    return candidates[:count]


def monotone_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a sampled signal into increasing positive and negative variation parts."""

    samples = np.asarray(samples, dtype=float)
    deltas = np.diff(samples, prepend=samples[0])
    positive = np.cumsum(np.clip(deltas, 0.0, None))
    negative = np.cumsum(np.clip(-deltas, 0.0, None))

    def normalize(values: np.ndarray) -> np.ndarray:
        span = values[-1] - values[0]
        if np.abs(span) < 1e-12:
            return np.linspace(0.0, 1.0, len(values))
        return (values - values[0]) / span

    return normalize(positive), normalize(negative)


@dataclass
class PaperLikeActivation:
    """Finite, runnable analogue of the paper's constructive activation.

    The paper defines a strictly increasing sigmoidal activation by stitching
    polynomial pieces on intervals of the form [4n, 4n+2] and their mirrored
    negative counterparts. This class implements a finite version of that idea:
    each interval is assigned a monotone shape derived from a polynomial split
    into increasing positive and negative parts, then the pieces are stitched
    together into one global strictly increasing lookup table.
    """

    polynomial_count: int = 8
    samples_per_interval: int = 128
    tail_scale: float = 1.5

    def __post_init__(self) -> None:
        self.polynomials = paper_like_polynomials(self.polynomial_count)
        self.x_table, self.y_table = self._build_lookup_table()

    def _build_lookup_table(self) -> tuple[np.ndarray, np.ndarray]:
        x_points: list[float] = []
        y_points: list[float] = []

        current_value = -0.96
        step = 1.92 / (2 * self.polynomial_count)
        local_grid = np.linspace(-1.0, 1.0, self.samples_per_interval)

        def append_interval(x_start: float, x_end: float, shape: np.ndarray) -> None:
            nonlocal current_value
            x_interval = np.linspace(x_start, x_end, self.samples_per_interval)
            y_interval = current_value + step * shape
            if x_points and np.isclose(x_interval[0], x_points[-1]):
                x_interval = x_interval[1:]
                y_interval = y_interval[1:]
            x_points.extend(x_interval.tolist())
            y_points.extend(y_interval.tolist())
            current_value = y_interval[-1]

        # Negative side first, from far left toward the origin.
        for index in reversed(range(self.polynomial_count)):
            polynomial = self.polynomials[index]
            sampled = polynomial(local_grid)
            positive_shape, negative_shape = monotone_split(sampled)
            interval_start = -(4 * index + 2)
            interval_end = -(4 * index)
            append_interval(interval_start, interval_end, negative_shape)

        # Positive side, from the origin toward the far right.
        for index, polynomial in enumerate(self.polynomials):
            sampled = polynomial(local_grid)
            positive_shape, negative_shape = monotone_split(sampled)
            interval_start = 4 * index
            interval_end = 4 * index + 2
            append_interval(interval_start, interval_end, positive_shape)

        return np.asarray(x_points), np.asarray(y_points)

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        scalar_input = values.ndim == 0
        values = np.atleast_1d(values)

        left_x = self.x_table[0]
        right_x = self.x_table[-1]
        left_y = self.y_table[0]
        right_y = self.y_table[-1]

        result = np.interp(np.clip(values, left_x, right_x), self.x_table, self.y_table)

        left_mask = values < left_x
        if np.any(left_mask):
            result[left_mask] = left_y - (left_y + 1.0) * np.exp((values[left_mask] - left_x) / self.tail_scale)

        right_mask = values > right_x
        if np.any(right_mask):
            result[right_mask] = right_y + (1.0 - right_y) * (1.0 - np.exp(-(values[right_mask] - right_x) / self.tail_scale))

        if scalar_input:
            return np.asarray(result[0])
        return result


def fit_shifted_two_neuron_model(
    target_fn,
    x_grid: np.ndarray,
    activation: PaperLikeActivation,
    shifts: Iterable[float],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit the paper-style 1D model c0 + c1*k(-x-d) + c2*k(x+d)."""

    y = target_fn(x_grid)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None

    for shift in shifts:
        basis = np.column_stack(
            [
                np.ones_like(x_grid),
                activation(-x_grid - shift),
                activation(x_grid + shift),
            ]
        )
        coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
        prediction = basis @ coeffs
        max_error = float(np.max(np.abs(y - prediction)))
        candidate = (max_error, float(shift), coeffs, prediction)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    return best


def build_fixed_direction_features_2d(
    points: np.ndarray,
    activation: PaperLikeActivation,
    shifts: Iterable[float],
) -> np.ndarray:
    """Build fixed-direction features for a 2D approximation demo.

    The feature set follows the paper's fixed-direction spirit by using a small
    collection of unit vectors and output-only fitting.
    """

    directions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0] / np.sqrt(2.0),
            [1.0, -1.0] / np.sqrt(2.0),
            [-1.0, 1.0] / np.sqrt(2.0),
            [-1.0, -1.0] / np.sqrt(2.0),
        ]
    )

    features = [np.ones(points.shape[0])]
    projected = points @ directions.T
    for shift in shifts:
        for column in range(projected.shape[1]):
            features.append(activation(projected[:, column] + shift))

    return np.column_stack(features)


def target_1d(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x**3 - x)


def target_2d(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return 0.35 * np.sin(np.pi * x) * np.cos(np.pi * y) + 0.25 * x * y + 0.15 * x**2 - 0.1 * y


def run_1d_demo(activation: PaperLikeActivation) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 500)
    shifts = np.linspace(0.0, 4.0, 33)
    max_error, best_shift, coeffs, prediction = fit_shifted_two_neuron_model(target_1d, x, activation, shifts)
    return x, target_1d(x), max_error, best_shift, prediction


def run_2d_demo(activation: PaperLikeActivation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 61)
    x_grid, y_grid = np.meshgrid(axis, axis)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    shifts = np.linspace(-1.25, 1.25, 9)
    design = build_fixed_direction_features_2d(points, activation, shifts)
    target = target_2d(points)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coeffs

    return x_grid, target.reshape(x_grid.shape), prediction.reshape(x_grid.shape)


def plot_results(
    x_1d: np.ndarray,
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    x_grid_2d: np.ndarray,
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    output_prefix: str = "uap_constructive_demo",
    show: bool = True,
) -> None:
    error_1d = np.abs(y_true_1d - y_pred_1d)
    error_2d = np.abs(y_true_2d - y_pred_2d)

    fig_1d, axes_1d = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes_1d[0].plot(x_1d, y_true_1d, label="target", linewidth=2)
    axes_1d[0].plot(x_1d, y_pred_1d, label="two-neuron approximation", linewidth=2)
    axes_1d[0].set_title("1D paper-inspired constructive approximation")
    axes_1d[0].legend()
    axes_1d[0].grid(alpha=0.3)

    axes_1d[1].plot(x_1d, error_1d, label="absolute error", color="crimson")
    axes_1d[1].set_xlabel("x")
    axes_1d[1].legend()
    axes_1d[1].grid(alpha=0.3)
    fig_1d.tight_layout()
    fig_1d.savefig(f"{output_prefix}_1d.png", dpi=160, bbox_inches="tight")

    fig_2d, axes_2d = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    extent = [-1, 1, -1, 1]
    plots = [
        (y_true_2d, "Target"),
        (y_pred_2d, "Approximation"),
        (error_2d, "Absolute error"),
    ]
    for axis, (image, title) in zip(axes_2d, plots):
        im = axis.imshow(image, origin="lower", extent=extent, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig_2d.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig_2d.savefig(f"{output_prefix}_2d.png", dpi=160, bbox_inches="tight")

    if show:
        plt.show()


def export_activation_experiment(
    activation: PaperLikeActivation,
    output_dir: Path,
    sample_count: int = 2000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min = float(activation.x_table[0])
    x_max = float(activation.x_table[-1])
    x_samples = np.linspace(x_min, x_max, sample_count)
    y_samples = activation(x_samples)

    samples = np.column_stack([x_samples, y_samples])
    np.savetxt(
        output_dir / "activation_samples.csv",
        samples,
        delimiter=",",
        header="x,activation",
        comments="",
    )

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(x_samples, y_samples, label="constructed activation", linewidth=2)
    axis.plot(activation.x_table, activation.y_table, label="stitching table", linewidth=1, alpha=0.8)
    axis.set_title("Algorithmically constructed activation")
    axis.set_xlabel("x")
    axis.set_ylabel("activation(x)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "activation_profile.png", dpi=160, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Algorithmically construct a paper-inspired strictly increasing activation "
            "and export a small reproducible approximation experiment."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where CSV and PNG outputs will be written.",
    )
    parser.add_argument(
        "--polynomial-count",
        type=int,
        default=8,
        help="How many polynomials to stitch into the activation construction.",
    )
    parser.add_argument(
        "--samples-per-interval",
        type=int,
        default=128,
        help="Number of samples used per stitched interval.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening matplotlib windows after generating the figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation = PaperLikeActivation(
        polynomial_count=args.polynomial_count,
        samples_per_interval=args.samples_per_interval,
    )

    x_1d, y_true_1d, max_error, best_shift, y_pred_1d = run_1d_demo(activation)
    print("1D best shift =", best_shift)
    print("1D max absolute error =", max_error)

    x_grid_2d, y_true_2d, y_pred_2d = run_2d_demo(activation)
    max_error_2d = float(np.max(np.abs(y_true_2d - y_pred_2d)))
    print("2D max absolute error =", max_error_2d)

    export_activation_experiment(activation, args.output_dir)
    plot_results(
        x_1d,
        y_true_1d,
        y_pred_1d,
        x_grid_2d,
        y_true_2d,
        y_pred_2d,
        show=not args.no_show,
    )

    if args.no_show:
        plt.close("all")


if __name__ == "__main__":
    main()