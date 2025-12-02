import numpy as np
import matplotlib.pyplot as plt
from graph_laplacian import image_to_laplacian
from subspace_iteration import run_experiment, run_matrix_experiment
from visualization import show_image_and_segmentation, plot_memory_usage, plot_runtime

# def show(img, title=""):
#     plt.figure(figsize=(3,3))
#     plt.imshow(img, cmap="gray", vmin=0, vmax=255)
#     plt.title(title)
#     plt.axis("off")

H1, W1 = 32, 32
img1 = np.zeros((H1, W1), dtype=np.uint8)
img1[:, :W1//2] = 50      # dark left half
img1[:, W1//2:] = 200     # bright right half

H3, W3 = 40, 40
img2 = np.zeros((H3, W3), dtype=np.uint8)
img2[:H3//2, :W3//2]     = 60
img2[:H3//2, W3//2:]     = 120
img2[H3//2:, :W3//2]     = 180
img2[H3//2:, W3//2:]     = 240

def diagonal_split(h=48, w=48):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if j > i:
                img[i, j] = 200
            else:
                img[i, j] = 50
    return img

img_diagonal = diagonal_split(48, 48)

def ring(h=64, w=64, r_inner=10, r_outer=20):
    yy, xx = np.indices((h, w))
    cx, cy = w//2, h//2
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    img = np.zeros((h, w), dtype=np.uint8)
    img[(dist > r_inner) & (dist < r_outer)] = 220
    return img

img_ring = ring(64, 64, 10, 20)

images = [img1, img2, img_diagonal, img_ring]
names = ["Half-split", "Quadrants", "Diagonal Split", "Ring Pattern"]

plt.figure(figsize=(14, 6))

for idx, (img, name) in enumerate(zip(images, names), start=1):
    H, W = img.shape
    plt.subplot(1, 4, idx)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{name}\n{H}×{W}")
    plt.axis("off")

plt.tight_layout()
plt.show()

L_1, labels = image_to_laplacian(
    img1,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=True,
    n_segments=20, # 20, 40, 60, 80, 90       
    use_superpixels=True
)

if hasattr(L_1, "toarray"):
    L_1 = L_1.toarray()
# print(L_1.shape[0])
L_2, labels = image_to_laplacian(
    img2,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=True,
    n_segments=40,       
    use_superpixels=True
)

if hasattr(L_2, "toarray"):
    L_2 = L_2.toarray()

L_3, labels = image_to_laplacian(
    img_diagonal,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=True,
    n_segments=60,       
    use_superpixels=True
)

if hasattr(L_3, "toarray"):
    L_3 = L_3.toarray()

L_4, labels = image_to_laplacian(
    img_ring,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=True,
    n_segments=100,       
    use_superpixels=True
)

if hasattr(L_4, "toarray"):
    L_4 = L_4.toarray()

L_5, labels = image_to_laplacian(
    img_ring,
    k=10,
    sigma=None,
    normalized=True,
    return_superpixel_labels=True,
    n_segments=200,       
    use_superpixels=True
)

if hasattr(L_5, "toarray"):
    L_5 = L_5.toarray()

def count_eigenvalues(L):
    if hasattr(L, "toarray"):
        L = L.toarray()
    return L.shape[0]

num_eigenvalues_1 = count_eigenvalues(L_1)
num_eigenvalues_2 = count_eigenvalues(L_2)
num_eigenvalues_3 = count_eigenvalues(L_3)
num_eigenvalues_4 = count_eigenvalues(L_4)
num_eigenvalues_5 = count_eigenvalues(L_5)

num_eigenvalues = {
    "img1": num_eigenvalues_1,
    "img2": num_eigenvalues_2,
    "diagonal": num_eigenvalues_3,
    "rings_100seg": num_eigenvalues_4,
    "rings_200seg": num_eigenvalues_5
}
print(num_eigenvalues)


results_img1 = run_experiment(
    img1,
    k_clusters=2,
    k_neighbors=10,
    max_iter=10,
    tol=1e-8,
    visualize=True,  # Set to True to see plots
    save_results=False,
    as_sparse=False,  # Use dense for small images
    true_img=False,  # No ground truth available
    n_segments=20
)

results_img2 = run_experiment(
    img2,
    k_clusters=4,
    k_neighbors=10,
    max_iter=75,
    tol=1e-8,
    visualize=True,  # Set to True to see plots
    save_results=False,
    as_sparse=False,  # Use dense for small images
    true_img=False,  # No ground truth available
    n_segments=40
)

results_diagonal = run_experiment(
    img_diagonal,
    k_clusters=2,
    k_neighbors=10,
    max_iter=375,
    tol=1e-8,
    visualize=True,  # Set to True to see plots
    save_results=False,
    as_sparse=False,  # Use dense for small images
    true_img=False,  # No ground truth available
    n_segments=60
)

results_rings = run_experiment(
    img_ring,
    k_clusters=3,
    k_neighbors=10,
    max_iter=500,
    tol=1e-8,
    visualize=True,  # Set to True to see plots
    save_results=False,
    as_sparse=False,  # Use dense for small images
    true_img=False,  # No ground truth available
    n_segments=100
)


results_ringl = run_experiment(
    ring(64, 64, 10, 20),
    k_clusters=3,
    k_neighbors=10,
    max_iter=500,
    tol=1e-8,
    visualize=True,  # Set to True to see plots
    save_results=False,
    as_sparse=False,  # Use dense for small images
    true_img=False,  # No ground truth available
    n_segments=200
)

all_results = {
    "img1": results_img1,
    "img2": results_img2,
    "diagonal": results_diagonal,
    "rings_100seg": results_rings,
    "rings_200seg": results_ringl
}


import matplotlib.pyplot as plt
import numpy as np

def extract_metric(all_results, metric):
    """
    Returns:
        solvers: list of solver names
        values: dict {experiment_name: [v_solver1, v_solver2, ...]}
    """
    # find solvers from first experiment
    first_exp = next(iter(all_results.values()))
    solvers = list(first_exp.keys())

    values = {}
    for exp_name, exp_results in all_results.items():
        vals = []
        for solver in solvers:
            v = exp_results[solver].get(metric, np.nan)
            vals.append(v)
        values[exp_name] = vals

    return solvers, values

def plot_metric_with_eig(all_results, eigen_counts, metric, title, log_scale=False):
    solvers, values = extract_metric(all_results, metric)

    exp_names = list(values.keys())
    num_exps = len(exp_names)
    x = np.arange(num_exps)

    plt.figure(figsize=(14, 6))

    # ----- BAR WIDTH -----
    width = 0.12  # adjust for spacing

    # ----- CREATE GROUPED BARS -----
    for i, solver in enumerate(solvers):
        solver_vals = [values[exp][i] for exp in exp_names]

        # Each solver gets an offset: center +/- (i * width)
        plt.bar(
            x + (i - len(solvers)/2) * width + width/2,
            solver_vals,
            width=width,
            label=solver
        )

    # ----- ADD EIGENVALUE COUNT ANNOTATIONS -----
    for idx, exp in enumerate(exp_names):
        eig_count = eigen_counts[exp]
        ypos = max(values[exp])  # highest bar for this experiment

        plt.text(
            x[idx],
            ypos * 1.02,
            f"eig = {eig_count}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    # ----- LABEL FIXING -----
    if metric == "runtime":
        metric = "Runtime (s)"
    elif metric == "peak_python_memory":
        metric = "Peak Python Memory (bytes)"

    plt.xticks(x, exp_names)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()



plot_metric_with_eig(
    all_results,
    num_eigenvalues,
    "runtime",
    "Runtime Comparison (with eigenvalue counts)",
    log_scale=True
)

plot_metric_with_eig(all_results, num_eigenvalues,
                     "peak_memory", "Peak Memory (with eigenvalue counts)")

plot_metric_with_eig(all_results, num_eigenvalues,
                     "peak_python_memory", "Peak Python Memory (with eigenvalue counts)", log_scale=True)

plot_metric_with_eig(all_results, num_eigenvalues,
                     "orthogonalization_loss", "Orthogonalization Loss (with eigenvalue counts)")
