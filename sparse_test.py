from subspace_iteration import run_experiments_on_images


k_arr = [5,5,5,5,5]

with open("images.txt", "r") as f:
    image_inputs = [line.strip() for line in f.readlines() if line.strip()]

# Image1 Train 151-175 (24063): 5-6 segs 6
# Image2 Test 26-50 (260058): 5-10 segs 7
# Image3: Test 76-100 (66053) 5-11 segs 8
# Image4: Test 76-100 (102061) 5-8 segs 5
# Image5: Test 76-100 (351093) 5-9 segs
all_results, agg = run_experiments_on_images(
            image_inputs,
            k_arr,
            k_neighbors=10,
            sigma=None,
            normalized_laplacian=True,
            max_iter=1000,
            tol=1e-10,
            visualize=True,        # now controls aggregate plots
            save_results=False,     # save aggregate plots to disk
            base_output_dir="results_batch",
            n_segments=1000
        )

import re

def _image_sort_key(img_id):
    """
    Extract numeric suffix from names like 'image1', 'image2', else fallback.
    """
    m = re.search(r'(\d+)', img_id)
    if m:
        return int(m.group(1))
    return img_id

import numpy as np

def get_metric_history(alg_result, metric):
    """
    Try to extract a per-iteration history for `metric` from one algorithm result dict.
    
    Priority:
      1. '<metric>_history' if exists and is sequence
      2. metric itself if it's a sequence
      3. metric itself if scalar -> return length-1 list
      4. otherwise: return None
    """
    # 1) <metric>_history
    hist_key = metric + "_history"
    if hist_key in alg_result:
        vals = alg_result[hist_key]
        if isinstance(vals, (list, tuple, np.ndarray)):
            return np.array(vals, dtype=float)

    # 2) metric as sequence
    if metric in alg_result:
        vals = alg_result[metric]
        if isinstance(vals, (list, tuple, np.ndarray)):
            return np.array(vals, dtype=float)
        # 3) metric as scalar
        if np.isscalar(vals):
            return np.array([float(vals)])

    # 4) not found
    return None

import matplotlib.pyplot as plt

def plot_metric_across_images(
    all_results,
    metric,
    skip_qr_for_runtime=True
):
    """
    Plot a single metric (e.g., runtime) across ALL images, 
    for ALL algorithms, on ONE graph.
    """

    # sort image IDs like image1, image2, ...
    image_ids = sorted(all_results.keys(), key=_image_sort_key)

    # find algorithm names from first image
    first_img_id = image_ids[0]
    algorithms = list(all_results[first_img_id].keys())

    plt.figure(figsize=(10, 6))

    for alg in algorithms:

        if metric == "runtime" and skip_qr_for_runtime and "QR Iteration" in alg:
            continue

        y_values = []

        for img_id in image_ids:
            alg_res = all_results[img_id][alg]

            # extract value or history
            hist = get_metric_history(alg_res, metric)

            if hist is None:
                y_values.append(np.nan)
            else:
                # if it's a history, plot LAST value (final iteration)
                y_values.append(hist[-1])

        plt.plot(range(len(image_ids)), y_values, marker="o", label=alg)

    plt.xticks(range(len(image_ids)), image_ids, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{metric} across all images")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # log-scale for runtime optionally
    if metric == "runtime":
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


metrics_to_plot = [
    "runtime",
    "peak_memory",
    "peak_python_memory",
    "orthogonalization_loss",
    "boundary_f1",
]

for m in metrics_to_plot:
    plot_metric_across_images(all_results, metric=m)

