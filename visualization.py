"""
Visualization utilities for experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_convergence(convergence_histories, algorithm_names, k=None, save_path=None):
    """
    Plot convergence history for multiple algorithms.

    Parameters
    ----------
    convergence_histories : list
        List of convergence histories (one per algorithm)
    algorithm_names : list
        Names of algorithms
    k : int, optional
        Number of eigenvalues to plot. If None, plots all.
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(
        len(convergence_histories),
        1,
        figsize=(10, 4 * len(convergence_histories)),
        sharex=True,
    )

    if len(convergence_histories) == 1:
        axes = [axes]

    for idx, (history, name) in enumerate(zip(convergence_histories, algorithm_names)):
        ax = axes[idx]

        if k is None:
            k_plot = len(history[0]) if history else 0
        else:
            k_plot = min(k, len(history[0]) if history else 0)

        for i in range(k_plot):
            values = [h[i] if i < len(h) else np.nan for h in history]
            ax.plot(values, label=f"λ{i+1}", alpha=0.7)

        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"{name} - Convergence History")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_segmentation(
    image, labels, image_shape, title="Segmentation Result", save_path=None
):
    """
    Plot image segmentation result.

    Parameters
    ----------
    image : ndarray
        Original image
    labels : ndarray
        Cluster labels for each pixel
    image_shape : tuple
        Shape of original image (height, width)
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Segmentation
    labels_reshaped = labels.reshape(image_shape[:2])
    axes[1].imshow(labels_reshaped, cmap="tab20")
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_comparison_results(results, save_path=None):
    """
    Create comparison plots for algorithm results.

    Parameters
    ----------
    results : dict
        Results dictionary from compare_algorithms
    save_path : str, optional
        Path to save figure
    """
    algorithms = list(results.keys())

    # Extract metrics
    runtimes = [results[alg]["runtime"] for alg in algorithms]
    n_iters = [results[alg]["n_iterations"] for alg in algorithms]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Runtime comparison
    axes[0].bar(algorithms, runtimes)
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title("Runtime Comparison")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Iterations comparison
    axes[1].bar(algorithms, n_iters)
    axes[1].set_ylabel("Number of Iterations")
    axes[1].set_title("Convergence Speed")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Accuracy comparison (if available)
    if "eigenvalue_relative_error" in results[algorithms[0]]:
        rel_errors = [results[alg]["eigenvalue_relative_error"] for alg in algorithms]
        axes[2].bar(algorithms, rel_errors)
        axes[2].set_ylabel("Relative Error")
        axes[2].set_title("Eigenvalue Accuracy")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(True, alpha=0.3, axis="y")
        axes[2].set_yscale("log")
    else:
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def create_report(results, image_shape=None, save_path=None):
    """
    Create a text report of experiment results.

    Parameters
    ----------
    results : dict
        Results dictionary from compare_algorithms
    image_shape : tuple, optional
        Shape of image for context
    save_path : str, optional
        Path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ALGORITHM COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    if image_shape:
        report_lines.append(f"Image shape: {image_shape}")
        report_lines.append(f"Number of pixels: {image_shape[0] * image_shape[1]}")
        report_lines.append("")

    for alg_name, result in results.items():
        report_lines.append(f"{alg_name}")
        report_lines.append("-" * 80)
        report_lines.append(f"  Runtime: {result['runtime']:.6f} seconds")
        report_lines.append(f"  Iterations: {result['n_iterations']}")
        report_lines.append(f"  Eigenvalues: {result['eigenvalues']}")

        if "eigenvalue_relative_error" in result:
            report_lines.append(
                f"  Eigenvalue relative error: {result['eigenvalue_relative_error']:.2e}"
            )

        if "eigenvector_mean_correlation" in result:
            report_lines.append(
                f"  Eigenvector mean correlation: {result['eigenvector_mean_correlation']:.4f}"
            )

        report_lines.append("")

    report_text = "\n".join(report_lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)

    print(report_text)
    return report_text
