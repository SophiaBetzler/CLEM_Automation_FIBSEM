import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def estimate_rigid_transform_fixed_scale(src, tgt, scale=1.0):
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    src = src * scale
    src_centroid = np.mean(src, axis=1, keepdims=True)
    tgt_centroid = np.mean(tgt, axis=1, keepdims=True)

    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    H = src_centered @ tgt_centered.T
    U, _, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T

    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T

    t = tgt_centroid - Rmat @ src_centroid
    return Rmat, t.squeeze()

def plot_alignment_result(src_pts, tgt_pts, R, t, pairing, inliers, scale=1.0):
    src_transformed = R @ (src_pts * scale) + t[:, None]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Track matched points
    matched_src = []
    matched_tgt = []
    unmatched_src_indices = set(range(src_pts.shape[1]))
    unmatched_tgt_indices = set(range(tgt_pts.shape[1]))

    for idx, (i, j) in enumerate(pairing):
        if inliers[idx]:
            matched_src.append(src_transformed[:, i])
            matched_tgt.append(tgt_pts[:, j])
            ax.plot([src_transformed[0, i], tgt_pts[0, j]],
                    [src_transformed[1, i], tgt_pts[1, j]], 'k--', alpha=0.5)
        unmatched_src_indices.discard(i)
        unmatched_tgt_indices.discard(j)

    # Plot matched
    if matched_src:
        matched_src = np.array(matched_src).T
        matched_tgt = np.array(matched_tgt).T
        ax.scatter(*matched_src, c='r', label='Matched Source')
        ax.scatter(*matched_tgt, c='b', label='Matched Target')

    # Plot unmatched
    if unmatched_src_indices:
        unmatched_src = src_transformed[:, list(unmatched_src_indices)]
        ax.scatter(*unmatched_src, c='gray', alpha=0.5, label='Unmatched Source')

    if unmatched_tgt_indices:
        unmatched_tgt = tgt_pts[:, list(unmatched_tgt_indices)]
        ax.scatter(*unmatched_tgt, c='lightblue', alpha=0.5, label='Unmatched Target')

    ax.set_title("Best Alignment with Inliers and Unmatched Points")
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def match_points_fixed_scale_brute(
    src_pts, tgt_pts,
    scale=1.2, min_points=3,
    top_n=2500, shortest_as_source=False
):
    src_pts = np.asarray(src_pts, dtype=np.float64)
    tgt_pts = np.asarray(tgt_pts, dtype=np.float64)

    D, Ns = src_pts.shape
    Nt = tgt_pts.shape[1]
    assert D == tgt_pts.shape[0]

    if shortest_as_source:
        if Ns > Nt:
            src_pts, tgt_pts = tgt_pts, src_pts
            Ns, Nt = Nt, Ns
            print("[INFO] Swapped: smaller set is now source.")
    else:
        if Ns < Nt:
            src_pts, tgt_pts = tgt_pts, src_pts
            Ns, Nt = Nt, Ns
            print("[INFO] Swapped: larger set is now source.")

    candidate_models = []

    combos = list(itertools.combinations(range(Ns), min_points))
    perms = list(itertools.permutations(range(Nt), min_points))

    total = len(combos) * len(perms)
    print(f"[INFO] Total model hypotheses to test: {total}")

    progress = 0
    for i_idx, i_combo in enumerate(combos):
        for j_idx, j_combo in enumerate(perms):
            progress += 1
            if progress % 1000 == 0 or progress == 1:
                print(f"[STATUS] Processing model {progress}/{total} ({(progress/total)*100:.1f}%)")

            src_sel = src_pts[:, i_combo]
            tgt_sel = tgt_pts[:, j_combo]

            Rmat, tvec = estimate_rigid_transform_fixed_scale(src_sel, tgt_sel, scale)
            src_all = Rmat @ (src_pts * scale) + tvec[:, None]

            if Ns <= Nt:
                permutations_to_use = itertools.permutations(range(Nt), Ns)
                for full_j in permutations_to_use:
                    tgt_all = tgt_pts[:, list(full_j)]
                    pairing = list(zip(range(Ns), list(full_j)))
                    error = np.mean(np.linalg.norm(src_all - tgt_all, axis=0)**2)
                    candidate_models.append((error, Rmat, tvec, pairing))
            else:
                permutations_to_use = itertools.permutations(range(Ns), Nt)
                for full_i in permutations_to_use:
                    src_subset = src_all[:, list(full_i)]
                    tgt_all = tgt_pts
                    pairing = list(zip(list(full_i), range(Nt)))
                    error = np.mean(np.linalg.norm(src_subset - tgt_all, axis=0)**2)
                    candidate_models.append((error, Rmat, tvec, pairing))

    print("[INFO] Sorting candidate models...")
    candidate_models.sort(key=lambda x: x[0])
    print("[INFO] Done.")
    return candidate_models[:top_n]


from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import numpy as np


def estimate_rigid_transform_fixed_scale(src, tgt, scale=1.0):
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    src = src * scale
    src_centroid = np.mean(src, axis=1, keepdims=True)
    tgt_centroid = np.mean(tgt, axis=1, keepdims=True)

    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    H = src_centered @ tgt_centered.T
    U, _, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T

    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T

    t = tgt_centroid - Rmat @ src_centroid
    return Rmat, t.squeeze()


def ransac_inliers(src_pts, tgt_pts, scale, R_seed, t_seed, inlier_thresh=10.0, min_samples=3, n_iter=100):
    best_inliers = []
    best_R, best_t = R_seed, t_seed
    Ns = src_pts.shape[1]
    Nt = tgt_pts.shape[1]

    min_common = min(Ns, Nt)
    if min_samples > min_common:
        raise ValueError("min_samples cannot be larger than the number of points in the smaller set.")

    for _ in range(n_iter):
        sample_idx = np.random.choice(min_common, min_samples, replace=False)
        src_sample = src_pts[:, sample_idx % Ns]
        tgt_sample = tgt_pts[:, sample_idx % Nt]

        R_new, t_new = estimate_rigid_transform_fixed_scale(src_sample, tgt_sample, scale)
        src_trans = R_new @ (src_pts * scale) + t_new[:, None]

        tree = cKDTree(tgt_pts.T)
        dists, nn_indices = tree.query(src_trans.T, k=1)
        inliers = np.where(dists < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R_new, t_new

    print(f"[DEBUG] RANSAC found {len(best_inliers)} inliers")
    return best_R, best_t, best_inliers

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def extract_transform_features(models):
    features = []
    for _, R, t, _ in models:
        angle = np.arctan2(R[1, 0], R[0, 0])
        scale = np.linalg.norm(R[:, 0])
        tx, ty = t[0], t[1]
        features.append([angle, tx, ty])
    return np.array(features)

def cluster_model_parameters(models, show_plot=True):
    print("[INFO] Clustering transformation parameters from candidate models...")
    features = extract_transform_features(models)
    clustering = DBSCAN(eps=0.1, min_samples=3).fit(features)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("[INFO] Cluster summary (label: count):")
    for label, count in zip(unique_labels, counts):
        label_str = f"Cluster {label}" if label != -1 else "Noise"
        print(f"  {label_str}: {count} models")

    # Optional: visualize rotation angle vs translation
    angles = features[:, 0]
    translations = features[:, 1:3]

    plt.figure(figsize=(6, 5))
    plt.scatter(angles, translations[:, 0], c=labels, cmap='tab10', s=20)
    plt.xlabel("Rotation Angle (rad)")
    plt.ylabel("Translation X")
    plt.title("Clustered Transformations: Angle vs Translation X")
    plt.tight_layout()
    plt.show()

        # Identify dominant cluster(s)
    label_counts = dict(zip(unique_labels, counts))
    dominant_label = max((label for label in unique_labels if label != -1), key=lambda l: label_counts[l], default=None)
    print(f"[INFO] Dominant cluster: {dominant_label} ({label_counts.get(dominant_label, 0)} models)")

    if show_plot:
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(angles, translations[:, 0], c=labels, cmap='tab10', s=20)
        plt.xlabel("Rotation Angle (rad)")
        plt.ylabel("Translation X")
        plt.title("Clustered Model Transformations")
        plt.legend(*scatter.legend_elements(), title="Cluster")
        plt.tight_layout()
        plt.show()

    return labels, dominant_label

def estimate_rigid_transform_fixed_scale(src, tgt, scale=1.0):
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    src = src * scale
    src_centroid = np.mean(src, axis=1, keepdims=True)
    tgt_centroid = np.mean(tgt, axis=1, keepdims=True)

    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    H = src_centered @ tgt_centered.T
    U, _, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T

    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T

    t = tgt_centroid - Rmat @ src_centroid
    return Rmat, t.squeeze()

def filter_models_by_cluster(models, labels, dominant_label=None, top_k=10):
    if dominant_label is None:
        print("[INFO] No dominant cluster found. Returning all models.")
        return models
    if dominant_label is not None:
        filtered = [model for model, label in zip(models, labels) if label == dominant_label]
    else:
        # Sort clusters by count and keep top_k labels (excluding noise)
        from collections import Counter
        label_counts = Counter(label for label in labels if label != -1)
        top_labels = [label for label, _ in label_counts.most_common(top_k)]
        filtered = [model for model, label in zip(models, labels) if label in top_labels]
    print(f"[INFO] Filtered {len(filtered)} models from dominant cluster.")
    return filtered



def evaluate_models_on_all_points(models, src_pts, tgt_pts, scale=1.0, inlier_thresh=10.0, plot=True,
                                  shortest_as_source=True, strategy="mean_residual", min_required_inliers=3):
    src_pts = np.asarray(src_pts, dtype=np.float64)
    tgt_pts = np.asarray(tgt_pts, dtype=np.float64)

    D, Ns = src_pts.shape
    Nt = tgt_pts.shape[1]
    assert D == tgt_pts.shape[0]

    source_swapped = False
    if shortest_as_source:
        if Ns > Nt:
            src_pts, tgt_pts = tgt_pts, src_pts
            Ns, Nt = Nt, Ns
            source_swapped = True
            print("[INFO] Swapped: smaller set is now source.")
    else:
        if Ns < Nt:
            src_pts, tgt_pts = tgt_pts, src_pts
            Ns, Nt = Nt, Ns
            source_swapped = True
            print("[INFO] Swapped: larger set is now source.")

    best_model = None
    best_score = np.inf
    best_metrics = {}

    for i, (error, R_seed, t_seed, pairing) in enumerate(models):
        if i % 2 == 0:
            print(f"[STATUS] Evaluating model {i + 1}/{len(models)}")

        idx_a, idx_b = zip(*pairing)
        idx_a = [i for i in idx_a if i < src_pts.shape[1]]
        idx_b = [j for j in idx_b if j < tgt_pts.shape[1]]

        if source_swapped:
            paired_src = tgt_pts[:, idx_b]
            paired_tgt = src_pts[:, idx_a]
        else:
            paired_src = src_pts[:, idx_a]
            paired_tgt = tgt_pts[:, idx_b]

        R, t, inlier_idxs = ransac_inliers(paired_src, paired_tgt, scale, R_seed, t_seed, inlier_thresh)
        if len(inlier_idxs) < min_required_inliers:
            print("[DEBUG] Skipping model: not enough RANSAC inliers")
            continue

        src_trans = R @ (src_pts * scale) + t[:, None]

        # Enforce unique target matches in RANSAC inlier collection
        used_tgt_ransac = set()
        ransac_pairings = []
        residuals = []
        for i in inlier_idxs:
            dists_i = [np.linalg.norm(src_trans[:, i] - tgt_pts[:, j]) for j in range(tgt_pts.shape[1])]
            j = np.argmin(dists_i)
            if dists_i[j] < inlier_thresh and j not in used_tgt_ransac:
                ransac_pairings.append((i, j))
                residuals.append(dists_i[j])
                used_tgt_ransac.add(j)
        used_tgt = set(j for _, j in ransac_pairings)
        pairing_final = list(ransac_pairings)

        print(f"[DEBUG] Final matches after RANSAC + greedy: {len(pairing_final)}")

        if len(pairing_final) < min_required_inliers:
            print("[DEBUG] Skipping model: not enough final matches")
            continue

        residuals = np.array(residuals)
        rmse = np.sqrt(np.mean(residuals ** 2))
        mean_residual = np.mean(residuals)

        if strategy == "rmse":
            score = rmse
        elif strategy == "inliers":
            score = -len(pairing_final)
        else:
            score = (-len(pairing_final), mean_residual)

        if best_model is None or score < best_score:
            best_model = (R, t, pairing_final, [True] * len(pairing_final))
            best_score = score
            best_metrics = {
                "inlier_count": len(pairing_final),
                "rmse": rmse,
                "mean_residual": mean_residual,
                "median_residual": np.median(residuals),
                "total": Ns
            }

    if best_model and plot:
        R_best, t_best, pairing_best, inliers_best = best_model
        plot_alignment_result(src_pts, tgt_pts, R_best, t_best, pairing_best, inliers_best, scale=scale)

    print("\n📊 Best Model Evaluation:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    return best_model, best_metrics


# 2D example
np.random.seed(0)
src = np.random.rand(2, 6)

# Apply known transform
true_scale = 1.1
true_angle = np.deg2rad(30)
R_true = np.array([[np.cos(true_angle), -np.sin(true_angle)],
                   [np.sin(true_angle),  np.cos(true_angle)]])
t_true = np.array([1.0, -0.5])
tgt = R_true @ (src * true_scale) + t_true[:, None]
tgt = tgt[:, 2:5]

# Add a bit of noise if you want
tgt += np.random.normal(0, 0.01, size=tgt.shape)


fl = np.array([[722.03466796875, 121.72842407226562], [802.1649780273438, 203.04808044433594], [880.6763305664062, 380.3256072998047], [871.7655944824219, 511.4419860839844], [791.6049194335938, 602.7266235351562], [716.484619140625, 674.1067504882812]])
print(np.shape(fl.T))
fib = np.array([[314.63824462890625, 468.5460205078125], [402.05517578125, 479.79083251953125], [494.89349365234375, 507.2642517089844], [604.644775390625, 515.518798828125], [762.989501953125, 501.46014404296875], [883.06103515625, 463.041748046875], [980.50048828125, 464.745849609375]])
print(np.shape(fib.T))
# Find best match with fixed scale
top_models = match_points_fixed_scale_brute(
    fib.T, fl.T,
    scale=0.8,         # your fixed scale guess
    min_points=3,
    top_n=2500
)
# best_model, best_metrics = evaluate_models_on_all_points(
#     top_models,
#     fib.T, fl.T,
#     scale=0.8,
#     inlier_thresh=250.0,  # distance threshold to count inliers
#     plot=True,
#     min_required_inliers=3
# )
# Step 1: Cluster the transformations
labels, dominant_label = cluster_model_parameters(top_models, show_plot=True)

# Step 2: Filter models using top K clusters (dominant_label is optional)
filtered_models = filter_models_by_cluster(top_models, labels, dominant_label=None, top_k=10)

# Step 3: Evaluate only the filtered models
best_model, best_metrics = evaluate_models_on_all_points(
    filtered_models,
    fib.T, fl.T,
    scale=0.8,
    inlier_thresh=250.0,
    plot=True,
    min_required_inliers=3
)