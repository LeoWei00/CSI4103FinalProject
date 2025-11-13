"""
Main experiment runner for comparing Subspace Iteration vs QR Algorithm
for spectral clustering in image segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
import os

# Import our modules
from graph_laplacian import image_to_laplacian, image_to_feature_vectors
from subspace_iteration_alg import standard_subspace_iteration, block_subspace_iteration
from qr_iteration import qr_iteration_partial
from lanczos import lanczos_iteration
from metrics import (
    compare_algorithms,
    spectral_clustering,
    compute_segmentation_metrics,
    measure_runtime,
)
from visualization import (
    plot_convergence,
    plot_segmentation,
    plot_comparison_results,
    create_report,
)


def load_image(image_input):
    """
    Load image from file path or numpy array.

    Parameters
    ----------
    image_input : str or ndarray
        Path to image file or numpy array

    Returns
    -------
    image : ndarray
        Image array
    """
    if isinstance(image_input, str):
        # Load from file
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = Image.open(image_input)
        image = np.array(img)
        return image
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise ValueError("image_input must be a file path (str) or numpy array")


def create_algorithm_wrappers(k, max_iter=1000, tol=1e-10):
    """
    Create algorithm wrapper functions for comparison.

    Parameters
    ----------
    k : int
        Number of eigenvectors to compute
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    algorithms : dict
        Dictionary of algorithm functions
    """

    def subspace_standard_wrapper(laplacian, k):
        eigenvals, eigenvecs, n_iter, history = standard_subspace_iteration(
            laplacian, k, max_iter=max_iter, tol=tol
        )
        return eigenvals, eigenvecs, n_iter, history

    def subspace_block_wrapper(laplacian, k):
        eigenvals, eigenvecs, n_iter, history = block_subspace_iteration(
            laplacian, k, max_iter=max_iter, tol=tol
        )
        return eigenvals, eigenvecs, n_iter, history

    def qr_wrapper(laplacian, k):
        # QR iteration doesn't return history, so we create a simple one
        eigenvals, eigenvecs, n_iter = qr_iteration_partial(
            laplacian, k, max_iter=max_iter, tol=tol
        )
        # Create dummy history (QR doesn't track intermediate values)
        history = [eigenvals] * n_iter
        return eigenvals, eigenvecs, n_iter, history

    def lanczos_wrapper(laplacian, k):
        eigenvals, eigenvecs, n_iter, history = lanczos_iteration(
            laplacian, k, max_iter=max_iter, tol=tol
        )
        return eigenvals, eigenvecs, n_iter, history

    algorithms = {
        "Subspace Iteration (Standard)": subspace_standard_wrapper,
        "Subspace Iteration (Block)": subspace_block_wrapper,
        "QR Iteration": qr_wrapper,
        "Lanczos": lanczos_wrapper,
    }

    return algorithms


def run_experiment(
    image_input,
    k_clusters=3,
    k_neighbors=10,
    sigma=None,
    normalized_laplacian=True,
    max_iter=1000,
    tol=1e-10,
    visualize=True,
    save_results=False,
    output_dir="results",
):
    """
    Run a complete experiment comparing algorithms.

    Parameters
    ----------
    image_input : str or ndarray
        Image file path or numpy array
    k_clusters : int
        Number of clusters for k-means
    k_neighbors : int
        Number of nearest neighbors for graph construction
    sigma : float, optional
        Bandwidth parameter for Gaussian weights
    normalized_laplacian : bool
        Whether to use normalized Laplacian
    max_iter : int
        Maximum iterations for algorithms
    tol : float
        Convergence tolerance
    visualize : bool
        Whether to show visualizations
    save_results : bool
        Whether to save results to files
    output_dir : str
        Directory to save results

    Returns
    -------
    results : dict
        Experiment results
    """
    print("=" * 80)
    print("SPECTRAL CLUSTERING ALGORITHM COMPARISON")
    print("=" * 80)
    print()

    # Load image
    print("Loading image...")
    image = load_image(image_input)
    image_shape = image.shape[:2]
    print(f"Image shape: {image_shape}")
    print(f"Number of pixels: {image_shape[0] * image_shape[1]}")
    print()

    # Construct graph Laplacian
    print("Constructing graph Laplacian...")
    print(f"  k-nearest neighbors: {k_neighbors}")
    print(f"  Normalized Laplacian: {normalized_laplacian}")
    laplacian = image_to_laplacian(
        image, k=k_neighbors, sigma=sigma, normalized=normalized_laplacian
    )
    print(f"  Laplacian shape: {laplacian.shape}")
    print()

    # Convert to dense if needed for some algorithms
    if hasattr(laplacian, "toarray"):
        laplacian_dense = laplacian.toarray()
    else:
        laplacian_dense = laplacian

    # Get reference solution using scipy (for accuracy comparison)
    print("Computing reference solution (scipy.linalg.eigh)...")
    from scipy.linalg import eigh

    eigenvals_ref, eigenvecs_ref = eigh(laplacian_dense)
    idx = np.argsort(eigenvals_ref)
    eigenvals_ref = eigenvals_ref[idx][:k_clusters]
    eigenvecs_ref = eigenvecs_ref[:, idx][:, :k_clusters]
    print()

    # Create algorithm wrappers
    algorithms = create_algorithm_wrappers(k_clusters, max_iter=max_iter, tol=tol)

    # Run comparison
    print("Running algorithm comparison...")
    results = compare_algorithms(
        laplacian_dense,
        k_clusters,
        algorithms,
        reference_eigenvals=eigenvals_ref,
        reference_eigenvecs=eigenvecs_ref,
    )
    print()

    # Perform spectral clustering with each method
    print("Performing spectral clustering...")
    segmentations = {}
    for alg_name, result in results.items():
        labels = spectral_clustering(result["eigenvectors"], k_clusters)
        segmentations[alg_name] = labels

        # Compute segmentation metrics
        seg_metrics = compute_segmentation_metrics(labels, image_shape=image_shape)
        result["segmentation_metrics"] = seg_metrics
        result["labels"] = labels

    print()

    # Create output directory if needed
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Visualizations
    if visualize or save_results:
        print("Generating visualizations...")

        # Convergence plots
        convergence_histories = [
            results[alg]["convergence_history"] for alg in algorithms.keys()
        ]
        if save_results:
            plot_convergence(
                convergence_histories,
                list(algorithms.keys()),
                k=k_clusters,
                save_path=os.path.join(output_dir, "convergence.png"),
            )
        if visualize:
            plot_convergence(
                convergence_histories, list(algorithms.keys()), k=k_clusters
            )

        # Segmentation results
        for alg_name, labels in segmentations.items():
            title = f"Segmentation: {alg_name}"
            save_path = None
            if save_results:
                save_path = os.path.join(
                    output_dir, f'segmentation_{alg_name.replace(" ", "_")}.png'
                )
            if visualize or save_results:
                plot_segmentation(
                    image, labels, image_shape, title=title, save_path=save_path
                )

        # Comparison plots
        if save_results:
            plot_comparison_results(
                results, save_path=os.path.join(output_dir, "comparison.png")
            )
        if visualize:
            plot_comparison_results(results)

    # Generate report
    print("Generating report...")
    report_path = None
    if save_results:
        report_path = os.path.join(output_dir, "report.txt")
    create_report(results, image_shape=image_shape, save_path=report_path)

    return results


def run_matrix_experiment(
    matrix,
    k,
    max_iter=1000,
    tol=1e-10,
    visualize=True,
    save_results=False,
    output_dir="results",
):
    """
    Run experiment with a matrix input (instead of image).

    Parameters
    ----------
    matrix : ndarray
        Input matrix (should be symmetric, e.g., graph Laplacian)
    k : int
        Number of eigenvectors to compute
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    visualize : bool
        Whether to show visualizations
    save_results : bool
        Whether to save results
    output_dir : str
        Output directory

    Returns
    -------
    results : dict
        Experiment results
    """
    print("=" * 80)
    print("MATRIX EIGENVALUE ALGORITHM COMPARISON")
    print("=" * 80)
    print()

    print(f"Matrix shape: {matrix.shape}")
    print(f"Computing {k} smallest eigenvalues/eigenvectors")
    print()

    # Get reference solution
    print("Computing reference solution (scipy.linalg.eigh)...")
    from scipy.linalg import eigh

    eigenvals_ref, eigenvecs_ref = eigh(matrix)
    idx = np.argsort(eigenvals_ref)
    eigenvals_ref = eigenvals_ref[idx][:k]
    eigenvecs_ref = eigenvecs_ref[:, idx][:, :k]
    print()

    # Create algorithm wrappers
    algorithms = create_algorithm_wrappers(k, max_iter=max_iter, tol=tol)

    # Run comparison
    print("Running algorithm comparison...")
    results = compare_algorithms(
        matrix,
        k,
        algorithms,
        reference_eigenvals=eigenvals_ref,
        reference_eigenvecs=eigenvecs_ref,
    )
    print()

    # Create output directory if needed
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Visualizations
    if visualize or save_results:
        print("Generating visualizations...")

        # Convergence plots
        convergence_histories = [
            results[alg]["convergence_history"] for alg in algorithms.keys()
        ]
        if save_results:
            plot_convergence(
                convergence_histories,
                list(algorithms.keys()),
                k=k,
                save_path=os.path.join(output_dir, "convergence.png"),
            )
        if visualize:
            plot_convergence(convergence_histories, list(algorithms.keys()), k=k)

        # Comparison plots
        if save_results:
            plot_comparison_results(
                results, save_path=os.path.join(output_dir, "comparison.png")
            )
        if visualize:
            plot_comparison_results(results)

    # Generate report
    print("Generating report...")
    report_path = None
    if save_results:
        report_path = os.path.join(output_dir, "report.txt")
    create_report(results, save_path=report_path)

    return results


def main():
    """Command-line interface for running experiments."""
    parser = argparse.ArgumentParser(
        description="Compare Subspace Iteration vs QR Algorithm for spectral clustering"
    )

    parser.add_argument(
        "input", type=str, help='Input: image file path or "matrix" for matrix input'
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of clusters/eigenvectors (default: 3)"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors for graph (default: 10)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Bandwidth parameter for Gaussian weights (default: auto)",
    )
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Use unnormalized Laplacian (default: normalized)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Maximum iterations (default: 1000)"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Convergence tolerance (default: 1e-10)",
    )
    parser.add_argument("--no-viz", action="store_true", help="Disable visualizations")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for saved results (default: results)",
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default=None,
        help="Path to .npy file containing matrix (for matrix input)",
    )

    args = parser.parse_args()

    if args.input.lower() == "matrix":
        # Matrix input mode
        if args.matrix_file is None:
            print("Error: --matrix-file required when input is 'matrix'")
            sys.exit(1)

        matrix = np.load(args.matrix_file)
        run_matrix_experiment(
            matrix,
            args.k,
            max_iter=args.max_iter,
            tol=args.tol,
            visualize=not args.no_viz,
            save_results=args.save,
            output_dir=args.output_dir,
        )
    else:
        # Image input mode
        run_experiment(
            args.input,
            k_clusters=args.k,
            k_neighbors=args.k_neighbors,
            sigma=args.sigma,
            normalized_laplacian=not args.unnormalized,
            max_iter=args.max_iter,
            tol=args.tol,
            visualize=not args.no_viz,
            save_results=args.save,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Example usage:")
        print("  python subspace_iteration.py image.jpg --k 3")
        print("  python subspace_iteration.py matrix --matrix-file laplacian.npy --k 5")
        print()
        print(
            "For interactive use, import and call run_experiment() or run_matrix_experiment()"
        )
        print()

        # You can also run a simple test here
        print("Running a simple test with a random matrix...")
        np.random.seed(42)
        n = 50
        # Create a random symmetric matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric
        # Make it positive semi-definite
        A = A @ A.T

        run_matrix_experiment(A, k=3, visualize=True, save_results=False)
    else:
        main()
