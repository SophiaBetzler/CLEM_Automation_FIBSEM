from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import itertools
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def match_points_fixed_scale_brute(
    src_pts, tgt_pts,
    scale=1.2, min_points=3,
    top_n=2500, shortest_as_source=False):

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

def apply_ransac_and_greedy(src_pts, tgt_pts, models, scale=1.0, inlier_thresh=10.0, min_required_inliers=3):
    best_model = None
    best_score = np.inf
    best_metrics = {}

    for i, (error, R_seed, t_seed, pairing) in enumerate(models):
        print(f"[DEBUG] Evaluating model {i + 1}/{len(models)}")
        idx_a, idx_b = zip(*pairing)
        paired_src = src_pts[:, list(idx_a)]
        paired_tgt = tgt_pts[:, list(idx_b)]

        # Apply seed transform directly
        R, t = R_seed, t_seed
        src_trans = R @ (src_pts * scale) + t[:, None]

        # Initial matches from paired inliers
        used_src = set(idx_a)
        used_tgt = set(idx_b)
        pairing_final = list(pairing)
        residuals = [np.linalg.norm(src_trans[:, i] - tgt_pts[:, j]) for i, j in pairing]

        # Greedy supplement
        pairs = [(i, j, np.linalg.norm(src_trans[:, i] - tgt_pts[:, j]))
                 for i in range(src_pts.shape[1])
                 for j in range(tgt_pts.shape[1])]
        pairs.sort(key=lambda x: x[2])

        for i, j, d in pairs:
            if d < inlier_thresh and i not in used_src and j not in used_tgt:
                pairing_final.append((i, j))
                residuals.append(d)
                used_src.add(i)
                used_tgt.add(j)

        if len(pairing_final) < min_required_inliers:
            continue

        residuals = np.array(residuals)
        rmse = np.sqrt(np.mean(residuals ** 2))
        mean_residual = np.mean(residuals)
        score = (-len(pairing_final), mean_residual)
        print(f"[DEBUG] Model score: inliers={len(pairing_final)}, mean_residual={mean_residual:.3f}, rmse={rmse:.3f}")
        if best_model is None or score < best_score:
            best_model = (R, t, pairing_final, [True] * len(pairing_final))
            best_score = score
            best_metrics = {
                "inlier_count": len(pairing_final),
                "rmse": rmse,
                "mean_residual": mean_residual,
                "median_residual": np.median(residuals),
                "total": src_pts.shape[1]
            }

            if best_model is None:
                print("[INFO] No valid model found after RANSAC + greedy matching.")
            if best_model:
                print("[INFO] Matched pairs (source_idx -> target_idx):")
            for i, j in pairing:
                print(f"  Source {i} -> Target {j}")
                print("[INFO] Matched pairs:")
            for i, j in pairing:
                print(f"  Source index {i} (src_pts) -> Target index {j} (tgt_pts)")
                if best_model:
                    print("[INFO] Matched coordinate pairs:")
                for i, j in pairing_final:
                    src_pt = src_pts[:, i]
                    tgt_pt = tgt_pts[:, j]
                    print(f"  Source {i} {tuple(src_pt)} -> Target {j} {tuple(tgt_pt)}")

        return best_model, best_metrics, [(src_pts[:, i], tgt_pts[:, j]) for i, j in pairing] if best_model else [], pairing_final if best_model else []

def run_full_fixed_scale_pipeline(src_pts, tgt_pts, candidate_models, scale=1.0, inlier_thresh=10.0, top_k=10, min_required_inliers=3):
    # Step 1: Cluster models
    labels, _ = cluster_model_parameters(candidate_models)

    # Step 2: Filter top-K clusters
    filtered_models = filter_models_by_cluster(candidate_models, labels, dominant_label=None, top_k=top_k)

    best_model, best_metrics, matched_coordinates, matched_indices = apply_ransac_and_greedy(
        src_pts, tgt_pts, filtered_models,
        scale=scale,
        inlier_thresh=inlier_thresh,
        min_required_inliers=min_required_inliers
    )

    # Step 4: Visualize result
    if best_model:
        R, t, pairing, inliers = best_model
        src_trans = R @ (src_pts * scale) + t[:, None]

        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, (i, j) in enumerate(pairing):
            if inliers[idx]:
                ax.plot([src_trans[0, i], tgt_pts[0, j]], [src_trans[1, i], tgt_pts[1, j]], 'k--', alpha=0.6)
        ax.scatter(src_trans[0], src_trans[1], c='red', label='Transformed Source')
        ax.scatter(tgt_pts[0], tgt_pts[1], c='blue', label='Target')
        ax.set_title("Best Model Alignment (Top Cluster)")
        ax.axis('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return best_model, best_metrics, matched_coordinates, matched_indices

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

def extract_transform_features(models):
    features = []
    for _, R, t, _ in models:
        angle = np.arctan2(R[1, 0], R[0, 0])
        scale = np.linalg.norm(R[:, 0])
        tx, ty = t[0], t[1]
        features.append([angle, tx, ty])
    return np.array(features)

def cluster_model_parameters(models, show_plot=True, plot_top_n_clusters=False, top_n=20):
    print("[INFO] Clustering transformation parameters from candidate models...")
    features = extract_transform_features(models)
    clustering = DBSCAN(eps=0.1, min_samples=3).fit(features)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("[INFO] Cluster summary (label: count):")
    for label, count in zip(unique_labels, counts):
        label_str = f"Cluster {label}" if label != -1 else "Noise"
        print(f"  {label_str}: {count} models")

    angles = features[:, 0]
    translations = features[:, 1:3]

    # Plot all clusters (no legend)
    if show_plot:
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(angles, translations[:, 0], c=labels, cmap='tab20', s=10)
        plt.xlabel("Rotation Angle (rad)")
        plt.ylabel("Translation X")
        plt.title("All Model Clusters")
        plt.tight_layout()
        plt.show()

    if plot_top_n_clusters:
        from collections import Counter
        label_counts = Counter(label for label in labels if label != -1)
        top_labels = [label for label, _ in label_counts.most_common(top_n)]
        mask = np.isin(labels, top_labels)

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(angles[mask], translations[:, 0][mask], c=labels[mask], cmap='tab20', s=10)
        plt.xlabel("Rotation Angle (rad)")
        plt.ylabel("Translation X")
        plt.title(f"Top {top_n} Clusters")
        plt.tight_layout()
        plt.show()

    label_counts = dict(zip(unique_labels, counts))
    dominant_label = max((label for label in unique_labels if label != -1), key=lambda l: label_counts[l], default=None)
    print(f"[INFO] Dominant cluster: {dominant_label} ({label_counts.get(dominant_label, 0)} models)")

    return labels, dominant_label

    label_counts = dict(zip(unique_labels, counts))
    dominant_label = max((label for label in unique_labels if label != -1), key=lambda l: label_counts[l], default=None)
    print(f"[INFO] Dominant cluster: {dominant_label} ({label_counts.get(dominant_label, 0)} models)")

    return labels, dominant_label

fl = np.array([[722.03466796875, 121.72842407226562], [802.1649780273438, 203.04808044433594], [880.6763305664062, 380.3256072998047], [871.7655944824219, 511.4419860839844], [791.6049194335938, 602.7266235351562], [716.484619140625, 674.1067504882812]])
fib = np.array([[314.63824462890625, 468.5460205078125], [402.05517578125, 479.79083251953125], [494.89349365234375, 507.2642517089844], [604.644775390625, 515.518798828125], [762.989501953125, 501.46014404296875], [883.06103515625, 463.041748046875], [980.50048828125, 464.745849609375]])

all_models = match_points_fixed_scale_brute(
    src_pts=fib.T,
    tgt_pts=fl.T,
    scale=0.8,
    min_points=3,
    shortest_as_source=False  # or True if needed
)

best_model, metrics, matched_coordinates, matched_indexes = run_full_fixed_scale_pipeline(
    src_pts=fib.T,
    tgt_pts=fl.T,
    candidate_models=all_models,
    scale=0.8,
    inlier_thresh=250.0
)

labels, dominant_label = cluster_model_parameters(
    all_models,
    show_plot=True,
    plot_top_n_clusters=True,  # enables second plot
    top_n=10                   # number of top clusters to highlight
)


# Full script with diagnostic plots added for each step

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import itertools
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# All functions from your original code go here, unchanged...
# Skipping for brevity

# -- Step 1: Initial point plot --
plt.figure(figsize=(6, 6))
plt.scatter(fib[:, 0], fib[:, 1], c='red', label='Source (fib)')
plt.scatter(fl[:, 0], fl[:, 1], c='blue', label='Target (fl)')
plt.title("Initial Point Sets")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Step 2: Brute-force matching
all_models = match_points_fixed_scale_brute(
    src_pts=fib.T,
    tgt_pts=fl.T,
    scale=0.8,
    min_points=3,
    shortest_as_source=False
)

# Plot histogram of candidate model errors
errors = [model[0] for model in all_models]
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50, color='gray')
plt.title("Distribution of Candidate Model Errors")
plt.xlabel("Mean Squared Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 3: Run full pipeline (includes clustering + best model selection)
best_model, metrics, matched_coordinates, matched_indexes = run_full_fixed_scale_pipeline(
    src_pts=fib.T,
    tgt_pts=fl.T,
    candidate_models=all_models,
    scale=0.8,
    inlier_thresh=250.0
)

# Step 4: Clustering parameters again with optional 3D and PCA plots
labels, dominant_label = cluster_model_parameters(
    all_models,
    show_plot=True,
    plot_top_n_clusters=True,
    top_n=10
)

# Optional 3D cluster plot
features = extract_transform_features(all_models)
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='tab20')
ax.set_xlabel("Angle (rad)")
ax.set_ylabel("Tx")
ax.set_zlabel("Ty")
ax.set_title("3D View of Transformation Clusters")
plt.tight_layout()
plt.show()

# PCA plot of clustered parameters
pca = PCA(n_components=2)
reduced = pca.fit_transform(features)
plt.figure(figsize=(6, 5))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10)
plt.title("Model Parameter Clustering (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# Step 5: Residual histogram and final pair visualization
if best_model:
    R, t, pairing, inliers = best_model
    src_trans = R @ (fib.T * 0.8) + t[:, None]

    residuals = [np.linalg.norm(src_trans[:, i] - fl.T[:, j]) for i, j in pairing]
    plt.hist(residuals, bins=30, color='skyblue')
    plt.title("Residuals of Final Matches")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Final matches overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    for src_pt, tgt_pt in matched_coordinates:
        ax.plot([src_pt[0], tgt_pt[0]], [src_pt[1], tgt_pt[1]], 'k--', alpha=0.5)
    ax.scatter([p[0] for p, _ in matched_coordinates], [p[1] for p, _ in matched_coordinates], c='red', label='Matched Source')
    ax.scatter([p[0] for _, p in matched_coordinates], [p[1] for _, p in matched_coordinates], c='blue', label='Matched Target')
    ax.set_title("Final Matched Pairs")
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

    # Optional: show transformed vs. original source
    plt.scatter(fib[:, 0], fib[:, 1], label='Original Source', alpha=0.5)
    plt.scatter(src_trans[0], src_trans[1], label='Transformed Source', alpha=0.8)
    plt.scatter(fl[:, 0], fl[:, 1], label='Target', alpha=0.8)
    plt.legend()
    plt.axis('equal')
    plt.title("Source Before/After Transformation")
    plt.grid(True)
    plt.show()

    print("\n==== FINAL MATCH SUMMARY ====")
    print(f"Best model inliers: {metrics['inlier_count']} / {metrics['total']}")
    print(f"Mean residual: {metrics['mean_residual']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print("Matched coordinate pairs:")
    for src_pt, tgt_pt in matched_coordinates:
        print(f"  Source: {src_pt} -> Target: {tgt_pt}")
