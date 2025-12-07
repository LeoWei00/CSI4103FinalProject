"""
Example usage of the subspace iteration comparison framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from subspace_iteration import run_experiment, run_matrix_experiment
from visualization import show_image_and_segmentation, plot_memory_usage, plot_runtime

plt.ion()

# Example 1: Run experiment with a numpy array (image)
print("Example 1: Image from numpy array")
print("-" * 80)

# Create a simple test image (grayscale)
# Using smaller size (40x40) to make QR iteration feasible
height, width = 40, 40
image = np.random.rand(height, width) * 255
image = image.astype(np.uint8)

# Add some structure
image[8:16, 8:16] = 200  # Bright square
image[24:32, 24:32] = 50  # Dark square

results = run_experiment(
    image,
    k_clusters=6,
    k_neighbors=10,
    max_iter=500,
    tol=1e-8,
    visualize=False,  # Set to True to see plots
    save_results=False,
    as_sparse=True,  # Use dense for small images
    true_img=False  # No ground truth available
)

for alg_name, result in results.items():
    labels = result.get("labels")
    if labels is not None:
        show_image_and_segmentation(image, labels, f"{alg_name} Segmentation")

plot_memory_usage(results)
plot_runtime(results)

print("\n" + "=" * 80)
print("Example 2: Matrix input")
print("-" * 80)

# Create a test symmetric matrix (simulating a graph Laplacian)
n = 50
np.random.seed(42)
A = np.random.randn(n, n)
A = (A + A.T) / 2  # Make symmetric
# Make it positive semi-definite (like a Laplacian)
A = A @ A.T

results = run_matrix_experiment(
    A,
    k=5,
    max_iter=200,
    tol=1e-10,
    visualize=False,  # Set to True to see plots
    save_results=False,
)

print("\n" + "=" * 80)
print("Example 3: Using individual algorithms")
print("-" * 80)

from graph_laplacian import image_to_laplacian
from subspace_iteration_alg import standard_subspace_iteration, block_subspace_iteration
from qr_iteration import qr_iteration_partial
from lanczos_qrimplicit import lanczos_practical_qr

# Create a small test matrix
n = 30
test_matrix = np.random.randn(n, n)
test_matrix = (test_matrix + test_matrix.T) / 2
test_matrix = test_matrix @ test_matrix.T

k = 3
mi = 1000

print("Running Standard Subspace Iteration...")
eigenvals_si, eigenvecs_si, n_iter_si, history_si = standard_subspace_iteration(
    test_matrix, k+1, max_iter=mi, tol=1e-10, skip_trivial=True
)
print(f"  Converged in {n_iter_si} iterations")
print(f"  Eigenvalues: {eigenvals_si}")

print("\nRunning QR Iteration...")
eigenvals_qr, eigenvecs_qr, n_iter_qr = qr_iteration_partial(
    test_matrix, k+1, max_iter=mi, tol=1e-10, skip_trivial=True
)
print(f"  Converged in {n_iter_qr} iterations")
print(f"  Eigenvalues: {eigenvals_qr}")

print("\nRunning QR + Lanczos...")
eigenvals_lanc_ir, eigenvecs_lanc_ir, n_iter_lanc_ir, history_lanc_ir = lanczos_practical_qr(
    test_matrix,
    k=k+1,
    m=(k+20),
    max_qr_iter=mi,
    tol=1e-10,
    skip_trivial=True
)
print(f"  Converged in {n_iter_lanc_ir} outer iterations")
print(f"  Eigenvalues: {eigenvals_lanc_ir}")

print("\nDone!")

plt.ioff()
plt.show()
