import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def estimate_similarity_transform_2d(B, A):
    """Estimate similarity transform (scale, rotation, translation) that aligns B to A."""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    norm_A = np.linalg.norm(A_centered)
    norm_B = np.linalg.norm(B_centered)
    scale = norm_A / norm_B

    A_unit = A_centered / norm_A
    B_unit = B_centered / norm_B

    H = B_unit.T @ A_unit
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_A - scale * (R @ centroid_B)
    B_aligned = (scale * (R @ B.T)).T + t
    return B_aligned, scale, R, t

def match_fiducials_nn(A, B, max_distance):
    """Match B to A using nearest neighbors and filter by distance."""
    tree = cKDTree(A)
    dists, indices = tree.query(B, k=1)

    matches = []
    matched_B = []
    matched_A = []
    unmatched_B = []
    unmatched_A = set(range(len(A)))

    for i_B, (idx, dist) in enumerate(zip(indices, dists)):
        if dist < max_distance:
            matches.append((i_B, idx))
            matched_B.append(i_B)
            matched_A.append(idx)
            unmatched_A.discard(idx)
        else:
            unmatched_B.append(i_B)

    return {
        "matches": matches,
        "B_indices": matched_B,
        "A_indices": matched_A,
        "unmatched_B": unmatched_B,
        "unmatched_A": list(unmatched_A)
    }

def icp_similarity_alignment(fiducials_A, fiducials_B, max_distance=5.0, max_iterations=10, tolerance=1e-4):
    """
    Iteratively aligns B to A with unknown correspondences.
    Returns aligned B and estimated transform.
    """
    A = np.asarray(fiducials_A)
    B = np.asarray(fiducials_B)
    B_aligned = B.copy()
    scale_total = 1.0
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for iteration in range(max_iterations):
        match_info = match_fiducials_nn(A, B_aligned, max_distance)

        if len(match_info["matches"]) < 3:
            print(f"Iteration {iteration}: Not enough matches to estimate transform.")
            break

        A_matched = A[match_info["A_indices"]]
        B_matched = B_aligned[match_info["B_indices"]]

        B_new, scale, R, t = estimate_similarity_transform_2d(B_matched, A_matched)

        # Update cumulative transform
        scale_total *= scale
        R_total = R @ R_total
        t_total = (scale * R @ t_total) + t

        delta = np.linalg.norm(B_new - B_aligned)
        B_aligned = B_new

        print(f"Iteration {iteration}: Δ={delta:.5f}, matches={len(match_info['matches'])}")
        if delta < tolerance:
            break

    return {
        "aligned_B": B_aligned,
        "scale": scale_total,
        "R": R_total,
        "t": t_total,
        "match_info": match_info
    }

def plot_fiducial_alignment(A, B_original, B_aligned, match_info):
    plt.figure(figsize=(8, 6))
    plt.scatter(A[:, 0], A[:, 1], c='blue', label='Reference A')
    plt.scatter(B_original[:, 0], B_original[:, 1], c='red', marker='x', label='Original B')
    plt.scatter(B_aligned[:, 0], B_aligned[:, 1], c='green', marker='+', label='Aligned B')

    for i_B, i_A in match_info["matches"]:
        plt.plot([B_aligned[i_B, 0], A[i_A, 0]],
                 [B_aligned[i_B, 1], A[i_A, 1]], 'k--', alpha=0.5)

    if match_info["unmatched_B"]:
        unmatched_coords = B_aligned[match_info["unmatched_B"]]
        plt.scatter(unmatched_coords[:, 0], unmatched_coords[:, 1],
                    c='black', label='Unmatched B', marker='x')

    if match_info["unmatched_A"]:
        unmatched_coords = A[match_info["unmatched_A"]]
        plt.scatter(unmatched_coords[:, 0], unmatched_coords[:, 1],
                    facecolors='none', edgecolors='cyan', label='Missing in B', marker='o')

    plt.title("Iterative Fiducial Matching (ICP-style)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define original points A
A = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Define known transform
theta = np.radians(30)
scale = 2.0
R_true = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
t_true = np.array([5, -3])

# Apply transform to A to generate B
B = (scale * (R_true @ A.T)).T + t_true

# Estimate back
B_aligned, scale_est, R_est, t_est = estimate_similarity_transform_2d(B, A)

print("Original scale:", scale)
print("Estimated scale:", scale_est)

print("Original translation:", t_true)
print("Estimated translation:", t_est)

print("Original rotation matrix:\n", R_true)
print("Estimated rotation matrix:\n", R_est)


# Run iterative registration
result = icp_similarity_alignment(A, B, max_distance=3.0)

# Show results
plot_fiducial_alignment(A, B, result["aligned_B"], result["match_info"])

# Print transform
print("Estimated scale:", result["scale"])
print("Estimated translation:", result["t"])
print("Estimated rotation matrix:\n", result["R"])
print("Matched indices (B ↔ A):", result["match_info"]["matches"])
print("Unmatched B indices:", result["match_info"]["unmatched_B"])
print("Missing in B (A indices):", result["match_info"]["unmatched_A"])
