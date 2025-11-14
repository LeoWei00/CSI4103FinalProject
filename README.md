# Subspace Iteration vs QR Algorithm for Spectral Clustering

A comparison framework for evaluating different eigenvalue algorithms in the context of spectral clustering for image segmentation.

## Overview

This project compares **Subspace Iteration** and **QR Algorithm** (along with Lanczos) for computing eigenvectors required for spectral clustering in image segmentation. The goal is to empirically compare these approaches in terms of:

- **Runtime** - Computational efficiency
- **Convergence Speed** - Number of iterations to convergence
- **Segmentation Quality** - Accuracy of resulting image segmentation

## Algorithms Implemented

1. **Standard Subspace Iteration** - Power method for multiple vectors
2. **Block Subspace Iteration** - More efficient variant for multiple eigenvectors
3. **QR Iteration** - Custom implementation with Wilkinson shift
4. **Lanczos Method** - Efficient for sparse matrices but does not use QR
5. **Lanczos + Implict QR** - Implments QR

## Project Structure

```
project/
├── subspace_iteration.py      # Main experiment runner
├── graph_laplacian.py          # Graph construction (k-NN, Laplacian)
├── subspace_iteration_alg.py   # Subspace iteration implementations
├── qr_iteration.py            # QR iteration implementation
├── lanczos.py                 # Lanczos method implementation
├── metrics.py                 # Evaluation metrics and comparison utilities
├── visualization.py          # Plotting and visualization functions
└── example_usage.py           # Example scripts
```

## Dependencies

- `numpy` - Numerical computations
- `scipy` - Linear algebra and sparse matrices
- `scikit-learn` - K-means clustering and k-NN
- `matplotlib` - Visualization
- `PIL` (Pillow) - Image loading

To install dependencies, use: 
```bash
py -m pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Run experiment with image file
python subspace_iteration.py image.jpg --k 3 --k-neighbors 10

# Run experiment wiht multiple image files (image.txt contains the image files on new lines)
python main.py images.txt --k 3 --save --output-dir results_batch

# Run experiment with matrix input
python subspace_iteration.py matrix --matrix-file laplacian.npy --k 5

# Save results and disable visualization
python subspace_iteration.py image.jpg --k 3 --save --no-viz
```

### Python API

```python
from subspace_iteration import run_experiment, run_matrix_experiment

# Image input
results = run_experiment(
    image_input="image.jpg",  # or numpy array
    k_clusters=3,
    k_neighbors=10,
    max_iter=1000,
    tol=1e-10,
    visualize=True
)

run_experiments_on_images(
    image_inputs,          # List of images
    k_clusters=3,
    k_neighbors=10,
    sigma=None,
    normalized_laplacian=True,
    max_iter=1000,
    tol=1e-10,
    visualize=False,        # now controls aggregate plots
    save_results=False,     # save aggregate plots to disk
    base_output_dir="results_batch",
)

# Matrix input
results = run_matrix_experiment(
    matrix=laplacian_matrix,
    k=5,
    max_iter=1000,
    tol=1e-10
)
```

### Example Script

```bash
python example_usage.py
```

## Key Features

- **Graph Laplacian Construction**: k-nearest neighbors with Gaussian weights
- **Multiple Algorithm Comparison**: Side-by-side evaluation of different methods
- **Comprehensive Metrics**: Runtime, convergence history, eigenvalue/eigenvector accuracy
- **Visualization**: Convergence plots, segmentation results, comparison charts
- **Flexible Input**: Supports both image files and numpy arrays

## Performance Notes

- **QR Iteration** is computationally expensive (O(n³) per iteration) and may be slow for large images
- **Subspace Iteration**, **Lanczos** and **Lanczos + Implict QR** are more efficient for large sparse matrices
- Recommended image sizes: 40×40 to 100×100 pixels for reasonable runtime
- For larger images, consider using sparse matrix operations or reducing k-neighbors

## Output

The framework generates:

- **Convergence plots** - Eigenvalue convergence history for each algorithm
- **Segmentation visualizations** - Original image and segmentation results
- **Comparison charts** - Runtime, iterations, and accuracy metrics
- **Text reports** - Detailed numerical results

## License

This project is for academic/research purposes.
