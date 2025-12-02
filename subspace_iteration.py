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
from collections import defaultdict
from skimage import io

# Import our modules
from graph_laplacian import image_to_laplacian, image_to_feature_vectors
from subspace_iteration_alg import standard_subspace_iteration, block_subspace_iteration
from qr_iteration import qr_iteration_partial
from lanczos import lanczos_iteration
from lanczos_qrimplicit import lanczos_practical_qr
from scipy.sparse.linalg import eigsh
from metrics import (
    compare_algorithms,
    spectral_clustering,
    compute_segmentation_metrics,
    measure_runtime,
    labels_to_pixel_map,
    boundary_f1,
)
from visualization import (
    plot_convergence,
    plot_segmentation,
    plot_comparison_results,
    create_report,
    plot_metric_across_images_scatter,
    create_aggregate_report,
    plot_f1score
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

def read_bsds300_seg(seg_path):
    """
    Parse BSDS300 .seg format (as in Berkeley segbench).
    Returns: seg (H, W) int32 label image, with labels starting from 1.
    """

    width = height = None
    quads = []  # (s, r, c1, c2) 0-based in file

    with open(seg_path, 'r') as f:
        # parse header
        while True:
            line = f.readline()
            if line == '':
                raise ValueError("Premature EOF before 'data'")
            line = line.strip()
            if line == 'data':
                break
            if line.startswith('width'):
                width = int(line.split()[1])
            elif line.startswith('height'):
                height = int(line.split()[1])

        if width is None or height is None:
            raise ValueError("Missing width/height header in .seg")

        # parse data: four integers per record
        import re
        data = f.read()
        nums = list(map(int, re.findall(r'-?\d+', data)))
        if len(nums) % 4 != 0:
            raise ValueError("Data length is not multiple of 4 in .seg")

        vals = np.array(nums, dtype=np.int64).reshape(-1, 4).T  # shape (4, N)
        # convert 0-based -> 1-based to match MATLAB code (+1)
        vals = vals + 1
        s_arr, r_arr, c1_arr, c2_arr = vals

        # MATLAB code uses seg = zeros(width,height); seg(c1:c2, r) = s; seg = seg';
        # After transpose, resulting shape becomes (height, width)
        seg = np.zeros((width, height), dtype=np.int32)
        for s, r, c1, c2 in zip(s_arr, r_arr, c1_arr, c2_arr):
            # clamp just in case
            c1 = max(1, min(c1, width))
            c2 = max(1, min(c2, width))
            r  = max(1, min(r,  height))
            seg[c1-1:c2, r-1] = s  # Python slice end is exclusive

        seg = seg.T  # transpose like MATLAB

        # sanity checks (mirror MATLAB)
        if seg.min() < 1:
            raise ValueError("Some pixel not assigned a segment (min<1).")
        # ensure labels are 1..K without gaps
        uniq = np.unique(seg)
        if len(uniq) != uniq.max():
            # not fatal, but keep consistent by remapping to 1..K
            mapping = {v: i+1 for i, v in enumerate(uniq)}
            seg = np.vectorize(mapping.get, otypes=[np.int32])(seg)

        return seg

def load_segmentation(seg_path):
    ext = os.path.splitext(seg_path)[1].lower()

    if ext == ".seg":
        seg = read_bsds300_seg(seg_path)
        return [seg.astype(np.int32)]
    return []
    
def create_algorithm_wrappers(n, k, max_iter=1000, tol=1e-10):
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
    if n//2 <= k:
        if(n < 100):
            pqr_m = k+20
        else:
            pqr_m = k+10
    else:
        if(n < 100):
            pqr_m = k+n//2
        else:
            pqr_m = n//2
    
    # print(k)
    def subspace_standard_wrapper(laplacian):
        # print(k)
        inner_k = k
        eigenvals, eigenvecs, n_iter, history = standard_subspace_iteration(
            laplacian, inner_k, max_iter=max_iter, tol=tol, skip_trivial=True
        )
        return eigenvals, eigenvecs, n_iter, history

    # def subspace_block_wrapper(laplacian):
    #     inner_k = k
    #     eigenvals, eigenvecs, n_iter, history = block_subspace_iteration(
    #         laplacian, inner_k, max_iter=max_iter, tol=tol, skip_trivial=True
    #     )
    #     return eigenvals, eigenvecs, n_iter, history

    # def lanczos_wrapper(laplacian, k=k):
    #     eigenvals, eigenvecs, n_iter, history = lanczos_iteration(
    #         laplacian, k, max_iter=max_iter, tol=tol
    #     )
    #     return eigenvals, eigenvecs, n_iter, history
    
    # def lanczos_ir_wrapper(laplacian, k):
    #     """
    #     Wrapper for implicitly restarted Lanczos.
    #     """
    #     vals, vecs = eigsh(laplacian, k=k, which="SM")
    #     # no iteration history, so fake it
    #     history = [vals.copy()]
    #     return vals, vecs, 1, history
    
    def lanczos_practical_qr_wrapper(laplacian):
        inner_k = k
        eigenvals, eigenvecs, n_iter, history = lanczos_practical_qr(
            laplacian, inner_k, m=pqr_m, max_qr_iter=max_iter, tol=1e-10, skip_trivial=True
        )
        return eigenvals, eigenvecs, n_iter, history
    
    def qr_wrapper(laplacian):
        inner_k = k
        # QR iteration doesn't return history, so we create a simple one
        eigenvals, eigenvecs, n_iter, history = qr_iteration_partial(
            laplacian, inner_k, max_iter=max_iter, tol=tol, skip_trivial=True
        )
        # Create dummy history (QR doesn't track intermediate values)
        return eigenvals, eigenvecs, n_iter, history

    algorithms = {
        "Subspace Iteration (Standard)": subspace_standard_wrapper,
        # "Subspace Iteration (Block)": subspace_block_wrapper,
        #"Lanczos": lanczos_wrapper,
        "Lanczos + QR": lanczos_practical_qr_wrapper,
        "QR Iteration": qr_wrapper
    }

    return algorithms


def run_experiment(
    image_input,
    k_clusters=3,
    k_neighbors=10,
    sigma=None,
    normalized_laplacian=True,
    return_superpixel_labels=True,
    use_superpixels=True,
    as_sparse=True,
    true_img = True,
    n_segments=800,
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
    if true_img:
        gt_label_img = load_segmentation(image_input.replace(".jpg", ".seg"))[0]
    else:
        gt_label_img = None
    image_shape = image.shape[:2]
    print(f"Image shape: {image_shape}")
    print(f"Number of pixels: {image_shape[0] * image_shape[1]}")
    print()

    # Construct graph Laplacian
    print("Constructing graph Laplacian...")
    print(f"  k-nearest neighbors: {k_neighbors}")
    print(f"  Normalized Laplacian: {normalized_laplacian}")
    laplacian, superpixel_labels = image_to_laplacian(
        image,
        k=k_neighbors,
        sigma=sigma,
        normalized=normalized_laplacian,
        use_superpixels=use_superpixels,
        return_superpixel_labels=return_superpixel_labels,
        n_segments=n_segments
    )
    print(f"  Laplacian shape: {laplacian.shape}")
    print()

    # Convert to dense if needed for some algorithms
    if not as_sparse:
        print(f"  Sparse storage: {as_sparse}")
        if hasattr(laplacian, "toarray"):
            laplacian_dense = laplacian.toarray()
        else:
            print("Error: laplacian is not sparse but has no toarray() method.")
            laplacian_dense = laplacian
        
        algorithms = create_algorithm_wrappers(laplacian_dense.shape[0], k_clusters+1, max_iter=max_iter, tol=tol)
        print("Running algorithm comparison...")
        results = compare_algorithms(
            laplacian_dense,
            k_clusters,
            algorithms
        )
    else:
        algorithms = create_algorithm_wrappers(laplacian.shape[0], k_clusters+1, max_iter=max_iter, tol=tol)
        results = compare_algorithms(
            laplacian,
            k_clusters,
            algorithms
        )

    # Create algorithm wrappers
    
    print()

    # Perform spectral clustering with each method
    print("Performing spectral clustering...")
    segmentations = {}
    for alg_name, result in results.items():
        labels = spectral_clustering(result["eigenvectors"], k_clusters)
        segmentations[alg_name] = labels

        px_labels = labels_to_pixel_map(superpixel_labels, labels)
        if gt_label_img is not None:
            result['boundary_f1'] = boundary_f1(px_labels, gt_label_img, tol=1)

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
                    image,
                    labels,
                    image_shape,
                    title=title,
                    save_path=save_path,
                    superpixel_labels=superpixel_labels,
                )
                
        if visualize or save_results:
            plot_f1score(results)

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

def run_experiments_on_images(
    image_inputs,
    k_clusters_list,
    k_neighbors=10,
    sigma=None,
    normalized_laplacian=True,
    max_iter=1000,
    tol=1e-10,
    visualize=True,
    save_results=False,
    base_output_dir="results_batch",
    n_segments=800
):
    """
    Run experiments on multiple images, where each image uses a *different*
    value of k_clusters taken from k_clusters_list.

    If k_clusters_list has fewer entries than images, the last k value is reused.
    """

    os.makedirs(base_output_dir, exist_ok=True)

    all_results = {}

    num_images = len(image_inputs)
    num_k = len(k_clusters_list)

    for idx, img_input in enumerate(image_inputs):

        # choose the correct k for this image
        if idx < num_k:
            k_clusters = k_clusters_list[idx]
        else:
            k_clusters = k_clusters_list[-1]   # reuse last value

        # readable name
        if isinstance(img_input, str):
            img_name = os.path.basename(img_input)
            img_id = os.path.splitext(img_name)[0]
        else:
            img_id = f"image_{idx}"

        print("\n" + "=" * 80)
        print(f"Running batch experiment for image: {img_id}")
        print(f" → using k_clusters = {k_clusters}")
        print("=" * 80)

        results = run_experiment(
            img_input,
            k_clusters=k_clusters,
            k_neighbors=k_neighbors,
            sigma=sigma,
            normalized_laplacian=normalized_laplacian,
            max_iter=max_iter,
            tol=tol,
            visualize=visualize,
            save_results=save_results,
            n_segments=n_segments
        )

        all_results[img_id] = results

    # -------------------------------------------
    # Aggregate metrics (unchanged)
    # -------------------------------------------
    aggregates = {}
    if all_results:
        first_img_id = next(iter(all_results.keys()))
        alg_names = list(all_results[first_img_id].keys())

        for alg_name in alg_names:
            metric_values = defaultdict(list)

            for img_id, res in all_results.items():
                alg_res = res[alg_name]
                seg_metrics = alg_res.get("segmentation_metrics", {})

                runtime = alg_res.get("runtime")
                if runtime is not None:
                    metric_values["runtime"].append(runtime)

                n_iter = alg_res.get("n_iterations")
                if n_iter is not None:
                    metric_values["n_iterations"].append(n_iter)

                for m_name, m_val in seg_metrics.items():
                    if np.isscalar(m_val):
                        metric_values[m_name].append(m_val)

            aggregates[alg_name] = {}
            for m_name, vals in metric_values.items():
                vals_arr = np.array(vals, dtype=float)
                aggregates[alg_name][m_name] = {
                    "values": vals,
                    "mean": float(vals_arr.mean()),
                    "std": float(vals_arr.std()),
                }

    # aggregate visualizations (unchanged)
    # if (visualize or save_results) and aggregates:
    #     if save_results:
    #         report_file = os.path.join(base_output_dir, "aggregate_report.txt")
    #         create_aggregate_report(
    #             aggregates,
    #             num_images=len(image_inputs),
    #             save_path=report_file
    #         )

    #     if visualize:
    #         print("\nGenerating aggregate plots across images...")
    #         key_metrics = ["runtime", "n_iterations", "f1", "ari"]
    #         for metric in key_metrics:
    #             if any(metric in aggregates[alg] for alg in aggregates):
    #                 scatter_path = (
    #                     os.path.join(base_output_dir, f"{metric}_per_image.png")
    #                     if save_results else None
    #                 )
    #                 plot_metric_across_images_scatter(
    #                     aggregates, metric, save_path=scatter_path
    #                 )

    return all_results, aggregates


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

    # Sort
    idx = np.argsort(eigenvals_ref)
    eigenvals_ref = eigenvals_ref[idx]
    eigenvecs_ref = eigenvecs_ref[:, idx]

    # Skip the trivial eigenvalue (0)
    eigenvals_ref = eigenvals_ref[1 : k + 1]
    eigenvecs_ref = eigenvecs_ref[:, 1 : k + 1]
    print()

    # Create algorithm wrappers
    algorithms = create_algorithm_wrappers(k+1, max_iter=max_iter, tol=tol)

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
        "input", type=str, help='Input: image file path, ".txt" list, or "matrix"'
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

    # -------------------- MATRIX MODE --------------------
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

    # -------------------- BATCH IMAGE MODE (.txt) --------------------
    elif args.input.lower().endswith(".txt"):
        # Treat input as a text file listing image paths
        list_path = args.input
        if not os.path.isfile(list_path):
            print(f"Error: batch list file not found: {list_path}")
            sys.exit(1)

        with open(list_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]

        # Ignore empty lines and comments
        image_inputs = [
            ln for ln in lines
            if ln and not ln.startswith("#")
        ]

        if not image_inputs:
            print("Error: no valid image paths found in the list file.")
            sys.exit(1)

        print(f"Found {len(image_inputs)} images in list file.")

        # Run batch experiments
        run_experiments_on_images(
            image_inputs,
            k_clusters=args.k,
            k_neighbors=args.k_neighbors,
            sigma=args.sigma,
            normalized_laplacian=not args.unnormalized,
            max_iter=args.max_iter,
            tol=args.tol,
            visualize=not args.no_viz,
            save_results=args.save,
            base_output_dir=args.output_dir,
        )

    # -------------------- SINGLE IMAGE MODE --------------------
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
