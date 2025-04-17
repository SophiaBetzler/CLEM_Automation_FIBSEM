import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class SpotburnMatching:

    def __init__(self, fib_spotburns, fl_spotburns, fib_scale, fl_scale):
        self.list_fib_spotburns = fib_spotburns
        self.list_fl_spotburns = fl_spotburns
        self.fib_scale = fib_scale
        self.fl_scale = fl_scale

    def build_affine_ransac_model(self, points_src, points_dst):
        """Build two RANSAC regressors: one for x, one for y"""
        model_x = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
        model_y = make_pipeline(PolynomialFeatures(1), RANSACRegressor())

        model_x.fit(points_src, points_dst[:, 0])
        model_y.fit(points_src, points_dst[:, 1])

        return model_x, model_y


    def align_points(self, points_src, model_x, model_y):
        """Apply fitted RANSAC regressors to align points"""
        x_aligned = model_x.predict(points_src)
        y_aligned = model_y.predict(points_src)
        return np.vstack([x_aligned, y_aligned]).T

    def estimate_similarity_transform(self, src, dst):
        """
        Estimate similarity transform (rotation, scale, translation) from src → dst.
        Returns: scale, rotation_angle_radians, translation_vector
        """
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = np.trace(R.T @ H) / np.sum(src_centered ** 2)
        angle = np.arctan2(R[1, 0], R[0, 0])
        translation = dst_mean - scale * (R @ src_mean)

        return scale, angle, translation

    def match_fiducials_with_alignment(self, distance_threshold=10.0, visualize=True):
        """
        Match two sets of 2D fiducial points from different views.
        Parameters:
            scale_1: tuple (dy, dx) - pixel sizes in microns for image 1
            scale_2: tuple (dy, dx) - pixel sizes in microns for image 2
            distance_threshold: maximum distance (in microns) to consider a valid match
            visualize: whether to plot the result

        Returns:
            matches: list of index pairs (i, j)
            unmatched_1: list of indices in points_1 with no match
            unmatched_2: list of indices in points_2 with no match
            aligned_points_2: aligned version of points_2 in top-view space
        """

        fl_spotburns_scaled = np.array(self.list_fl_spotburns) * np.array(self.fl_scale[::-1])  # [x, y] scaling
        fib_spotburns_scaled = np.array(self.list_fib_spotburns) * np.array(self.fib_scale[::-1])

        min_len = min(len(fl_spotburns_scaled), len(fib_spotburns_scaled))
        model_x, model_y = self.build_affine_ransac_model(fib_spotburns_scaled[:min_len], fl_spotburns_scaled[:min_len])
        aligned_fib_spotburns = self.align_points(fib_spotburns_scaled, model_x, model_y)

        scale, angle_rad, translation = self.estimate_similarity_transform(fib_spotburns_scaled, aligned_fib_spotburns)
        angle_deg = np.rad2deg(angle_rad)
        print(f"Estimated scale: {scale:.4f}")
        print(f"Estimated angle: {angle_deg:.2f} degrees")

        cost_matrix = cdist(fl_spotburns_scaled, aligned_fib_spotburns)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_1 = []
        unmatched_2 = set(range(len(self.list_fib_spotburns)))

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < distance_threshold:
                matches.append((i, j))
                unmatched_2.discard(j)
            else:
                unmatched_1.append(i)

        unmatched_2 = list(unmatched_2)

        # --- Step 4: Visualization ---
        if visualize:
            plt.figure(figsize=(8, 6))
            plt.title("Fiducial Matching: Top View (blue) vs Aligned Angled View (red)")
            plt.scatter(fl_spotburns_scaled[:, 0], fl_spotburns_scaled[:, 1], label="Image 1 (Top View)", c='blue')
            plt.scatter(aligned_fib_spotburns[:, 0], aligned_fib_spotburns[:, 1], label="Image 2 (Aligned View)", c='red')
            for i, j in matches:
                p1 = fl_spotburns_scaled[i]
                p2 = aligned_fib_spotburns[j]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=0.7)
            plt.gca().invert_yaxis()
            plt.xlabel("X (µm)")
            plt.ylabel("Y (µm)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return matches, unmatched_1, unmatched_2


#angled_view = [[313, 471], [401, 482], [496, 509], [605, 518], [764, 504], [883, 465], [980, 467], [1065, 495]]
#top_view = [[122, 721], [205, 801], [286, 861], [380, 880], [512, 871], [602, 789], [676, 718]]

angled_view = [[315, 469], [403, 480], [495, 507], [605, 515], [763, 501], [883, 463], [981, 465]]
top_view = [[122, 722], [203, 802], [380, 881], [511, 872], [603, 792], [674, 716]]

#[array([722.09655762, 122.24281311]), array([802.16497803, 203.04808044]), array([880.67633057, 380.3256073 ]), array([871.76559448, 511.44198608]), array([791.60491943, 602.72662354]), array([716.48461914, 674.10675049])]
#[(468.5460205078125, 314.63824462890625), (479.79083251953125, 402.05517578125), (507.2642517089844, 494.89349365234375), (515.518798828125, 604.644775390625), (501.46014404296875, 762.989501953125), (463.041748046875, 883.06103515625), (464.745849609375, 980.50048828125)]


spotburn_matching = SpotburnMatching(fib_spotburns=angled_view, fl_spotburns=top_view, fib_scale=(0.0651, 0.0651),
                                     fl_scale=(0.0774, 0.0774))

matches, unmatched_1, unmatched_2, = spotburn_matching.match_fiducials_with_alignment(distance_threshold=3,
                                                                                      visualize=True)

print("Matched indices:", matches)
print("Unmatched in top view:", unmatched_1)
print("Unmatched in angled view:", unmatched_2)

