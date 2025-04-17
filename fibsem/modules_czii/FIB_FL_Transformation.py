from typing import Any

import numpy as np
from numpy import ndarray, dtype
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
import xml.etree.ElementTree as ET
from tifffile import TiffFile
from scipy.spatial.transform import Rotation as R
from PIL import Image
import os


class Transformation:

    def __init__(self, fib_fiducials, fl_fiducials, fl_stack_name, fib_image_name, path):
        self.path = path
        self.fib_image, self.fib_scale = self.import_images(os.path.join(self.path, fib_image_name))
        self.fl_stack, self.fl_scale = self.import_images(os.path.join(self.path, fl_stack_name))
        print(np.shape(self.fl_stack))
        self.fib_fiducials = fib_fiducials
        self.fl_fiducials = fl_fiducials
        print(np.shape(fl_fiducials))
        print(np.shape(fib_fiducials))

######################################################
### Functions required to do the 3D Transformation ###
######################################################
    def quaternion_to_rotation_matrix(self, e):
        """
        Quaternion used to define rotation in 3D mathematically. Only first two columns are used in 3DCT.
        """
        e0, e1, e2, e3 = e
        R = np.array([
            [e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2,    2 * (e1 * e2 - e0 * e3),    2 * (e1 * e3 + e0 * e2)],
            [2 * (e1 * e2 + e0 * e3),       e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2,    2 * (e2 * e3 - e0 * e1)],
            [2 * (e1 * e3 - e0 * e2),      2 * (e2 * e3 + e0 * e1),     e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2]])
        return R

    def quaternion_constraint(self, params):
        """
        params vector has 7 elements: [e0, e1, e2, e3, scale, dx, dy]
        """
        e = params[0:4]
        return np.sum(e ** 2) - 1

    def transform_points_quaternion(self, e, s, d, X):
        """
        Function which transforms the values to create fib_predict
        """
        e = e / np.linalg.norm(e)
        d = np.asarray(d).reshape(2, 1)
        r_full = self.quaternion_to_rotation_matrix(e)
        r_proj: ndarray[Any, dtype[Any]] = r_full[:2, :]
        s_rx = s * (r_proj @ X)
        d_tile = d @ np.ones((1, X.shape[1]))
        return s_rx + d_tile

#################################################################################################
### Functions which fine tune the picked fiducial and verify which are used during the fitting
#################################################################################################

    def auto_tune_delta_and_outlier_thresh(self, res_norms, target_inlier_percent=0.8):
        """
        Automatically tune Huber delta and outlier threshold based on residual stats.
        """
        sorted_r = np.sort(res_norms)
        n = len(sorted_r)
        delta_index = int(target_inlier_percent * n)
        delta = sorted_r[min(delta_index, n - 1)]
        median = np.median(res_norms)
        mad = np.median(np.abs(res_norms - median)) + 1e-8
        # outlier_thresh = median + outlier_thresh * mad # if we want to also dynamically deteremine the outlier_thresh
        outlier_thresh = 3.0
        return delta, outlier_thresh

    def huber_loss_with_mask(self, Y_true, Y_pred):
        """
        Function penalizes transformed points with large error and fully removes points above a pre-defined
        'error' threshold.
        """
        residuals = Y_true - Y_pred
        res_norms = np.linalg.norm(residuals, axis=0)
        delta, outlier_thresh = self.auto_tune_delta_and_outlier_thresh(res_norms)

        median = np.median(res_norms)
        mad = np.median(np.abs(res_norms - median)) + 1e-8  # avoid div by 0
        threshold = median + outlier_thresh * mad
        inlier_mask = res_norms <= threshold

        residuals_inliers = residuals[:, inlier_mask]
        abs_r = np.abs(residuals_inliers)

        mask = abs_r <= delta
        loss = np.sum(np.where(mask, 0.5 * abs_r ** 2, delta * (abs_r - 0.5 * delta)))
        return loss, inlier_mask

    def scale_fl_voxels(self, x):
        """
        Scale 3D points (confocal / FL) by voxel spacing to get real-world micrometer units.
        """
        dz, dy, dx = self.fl_scale
        scale_matrix = np.diag([dx, dy, dz])
        return scale_matrix @ x

    def scale_fib_pixels(self, y):
        """
        Scale 2D FIB coordinates by pixel size to get real-world micrometer units.
        """
        dy, dx = self.fib_scale[1:]
        return np.array([y[0, :] * dx, y[1, :] * dy])

    def objective_with_huber(self, params):
        """
        Runs the quaternion-based similarity transform between 3D and 2D point sets,
        accounting for voxel/pixel spacing in both modalities and removes outliers using the huber_loss_with_mask
        function to calculate the total loss.
        - params: [e0, e1, e2, e3, s, dx, dy] gathered from a guess
        """
        e = params[0:4]
        s = params[4]
        d = params[5:7]
        total_loss = 0.0
        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            x_scaled = self.scale_fl_voxels(X)  # shape (3, n)
            y_scaled = self.scale_fib_pixels(Y)  # shape (2, n)
            y_pred = self.transform_points_quaternion(e, s, d, x_scaled)
            loss, _ = self.huber_loss_with_mask(y_scaled, y_pred)
            total_loss += loss
        return total_loss

    def least_squares_objective(self, params):
        """
        quaternion-based similarity transform between 3D and 2D point sets,
        accounting for voxel/pixel spacing in both modalities and defines the error according to the 3DCT method.
        """
        e = params[0:4]
        s = params[4]
        d = params[5:7]
        total_loss = 0.0
        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            x_scaled = self.scale_fl_voxels(X)  # shape (3, n)
            y_scaled = self.scale_fib_pixels(Y)  # shape (2, n)
            y_pred = self.transform_points_quaternion(e, s, d, x_scaled)
            residuals = y_scaled - y_pred
            total_loss += np.sum(residuals ** 2)
        return total_loss

    def estimate_similarity_transform_3d_to_2d(self, X, Y):
        """
        Estimate initial parameters (R, s, d) for 3D→2D projection using Procrustes (SVD) method.
        Inputs:
        - X: shape (3, N) → 3D source points (already scaled)
        - Y: shape (2, N) → 2D target points (already scaled)
        Returns:
        - q: quaternion representing rotation (len 4)
        - s: scale
        - d: translation (2D)
        """
        N = X.shape[1]
        mu_X = np.mean(X, axis=1, keepdims=True)
        mu_Y = np.mean(Y, axis=1, keepdims=True)
        Xc = X - mu_X
        Yc = Y - mu_Y
        A = Yc @ np.linalg.pinv(Xc)
        s = np.linalg.norm(A, ord='fro') / np.sqrt(2)
        R_2x3 = A / s
        u, _, vh = np.linalg.svd(R_2x3)
        R_full = np.eye(3)
        R_full[:2, :] = R_2x3
        rotation = R.from_matrix(R_full)
        q = rotation.as_quat()  # (x, y, z, w) format
        q = np.roll(q, 1)
        d = mu_Y - s * R_2x3 @ mu_X
        d = d.flatten()
        return q, s, d

    def guess_parameters_svd(self):
        """
        Creates the initial parameters used to fit the function.
        """
        print(self.fib_fiducials)
        print(np.shape(self.fib_fiducials))
        X_all = np.hstack([self.scale_fl_voxels(X) for X in self.fl_fiducials])
        Y_all = np.hstack([self.scale_fib_pixels(Y) for Y in self.fib_fiducials])
        q, s, d = self.estimate_similarity_transform_3d_to_2d(X_all, Y_all)
        return np.concatenate([q, [s], d])

    def guess_parameters(self, scale_range=(0.5, 2.0), trans_range=None):
        """
        Generates a random set of initial parameters:
        Returns:
        - params: numpy array of shape (7,) → [e0, e1, e2, e3, s, dx, dy]
        """
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        s = np.random.uniform(*scale_range)
        if trans_range is None:
            y_all = np.hstack(self.fib_fiducials)
            y_min = np.min(y_all, axis=1)
            y_max = np.max(y_all, axis=1)
            trans_range = [(y_min[0], y_max[0]), (y_min[1], y_max[1])]
        dx = np.random.uniform(*trans_range[0])
        dy = np.random.uniform(*trans_range[1])
        params = np.concatenate([q, [s], [dx, dy]])
        return params

        # Modify the run_transformation method to add plotting
        def run_transformation(self, params_init):
            """
            Function runs the two-step optimization process with visualization.
            """
            constraints = {'type': 'eq', 'fun': self.quaternion_constraint}
            loss_history = []

            def loss_callback(params):
                loss = self.least_squares_objective(params)
                loss_history.append(loss)

            # Visualize initial guess
            self.visualize_alignment(params_init, os.path.join(self.path, "initial_guess_overlay.png"),
                                     title="Initial Guess")

            # --- Robust (Huber) optimization ---
            result_robust = minimize(
                fun=self.objective_with_huber,
                x0=params_init,
                method='SLSQP',
                constraints=constraints,
                options={'disp': True}
            )
            params_first = result_robust.x
            self.visualize_alignment(params_first, os.path.join(self.path, "post_huber_overlay.png"),
                                     title="Post Huber Optimization")

            # Filter inliers for refinement
            e_first, s_first, d_first = params_first[0:4], params_first[4], params_first[5:7]
            fl_inliers, fib_inliers = [], []
            for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
                y_pred = self.transform_points_quaternion(e_first, s_first, d_first, X)
                _, mask = self.huber_loss_with_mask(Y, y_pred)
                fl_inliers.append(X[:, mask])
                fib_inliers.append(Y[:, mask])

            # --- Least squares refinement ---
            result_refined = minimize(
                fun=self.least_squares_objective,
                x0=params_first,
                method='SLSQP',
                constraints=constraints,
                options={'disp': True},
                callback=loss_callback
            )

            # Final params
            params_final = result_refined.x
            self.e_final, self.s_final, self.d_final = params_final[0:4], params_final[4], params_final[5:7]
            self.visualize_alignment(params_final, os.path.join(self.path, "final_alignment_overlay.png"),
                                     title="Final Alignment")

            # Loss curve plot
            if len(loss_history) > 1:
                plt.figure(figsize=(8, 4))
                plt.plot(loss_history, marker='o')
                plt.title("Least Squares Optimization Loss")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.path, "loss_curve.png"))
                plt.close()

            self.generate_all_overlays()
            self.plot_fiducial_alignment()

        # New helper method to visualize a given transform
        def visualize_alignment(self, params, save_path, title="Alignment"):
            e = params[0:4] / np.linalg.norm(params[0:4])
            s = params[4]
            d = params[5:7].reshape(2, 1)
            dy_fib, dx_fib = self.fib_scale[1:]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.fib_image, cmap='gray')

            for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
                x_scaled = self.scale_fl_voxels(X)
                y_scaled = self.scale_fib_pixels(Y)
                y_pred = self.transform_points_quaternion(e, s, d, x_scaled)

                y_scaled_pix = y_scaled[[0, 1]] / np.array([[dx_fib], [dy_fib]])
                y_pred_pix = y_pred[[0, 1]] / np.array([[dx_fib], [dy_fib]])

                ax.scatter(y_scaled_pix[0], y_scaled_pix[1], c='lime', label='FIB Fiducials', s=40)
                ax.scatter(y_pred_pix[0], y_pred_pix[1], c='red', marker='x', label='Predicted', s=30)

                for i in range(Y.shape[1]):
                    ax.plot([y_scaled_pix[0, i], y_pred_pix[0, i]],
                            [y_scaled_pix[1, i], y_pred_pix[1, i]],
                            'gray', linestyle='--', linewidth=1)

            ax.set_title(title)
            ax.axis('off')
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc='lower right')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    ########################################################################
### Create overlays and other outputs from the correlation process
########################################################################

    def generate_fl_voxel_coords(self, stack_shape):
        Z, Y, X = stack_shape
        dz, dy, dx = self.fl_scale
        z, y, x = np.meshgrid(
            np.arange(Z) * dz,
            np.arange(Y) * dy,
            np.arange(X) * dx,
            indexing='ij' )
        coords = np.stack([x, y, z], axis=0)  # shape: (3, Z, Y, X)
        return coords.reshape(3, -1)  # shape: (3, N)

    def plot_fiducial_alignment(self):
        """
        Plots overlay of registered FL fiducials and FIB fiducials on the FIB image.
        Also overlays transformed FL volume points for additional verification.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.fib_image, cmap='gray')

        e = self.e_final / np.linalg.norm(self.e_final)
        s = self.s_final
        d = self.d_final.reshape(2, 1)
        dy_fib, dx_fib = self.fib_scale[1:]

        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            x_scaled = self.scale_fl_voxels(X)  # in microns
            y_scaled = self.scale_fib_pixels(Y)  # in microns
            y_pred = self.transform_points_quaternion(e, s, d, x_scaled)
            y_scaled_pix = y_scaled[[0, 1]] / np.array([[dx_fib], [dy_fib]])
            y_pred_pix = y_pred[[0, 1]] / np.array([[dx_fib], [dy_fib]])

            ax.scatter(y_scaled_pix[0], y_scaled_pix[1], c='lime', label='Ground Truth', s=30)
            ax.scatter(y_pred_pix[0], y_pred_pix[1], c='red', label='Predicted', s=15)

        ax.set_title("Fiducial Registration Overlay")
        ax.axis('off')
        ax.legend()
        plt.savefig(os.path.join(self.path, 'Fiducial_registration_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        intensity_threshold = 0.1
        point_sampling = 0.25
        dz, dy, dx = self.fl_scale
        fl = self.fl_stack[0][1]
        indices = np.array(np.nonzero(fl > intensity_threshold))
        print("indices shape:", indices.shape)
        intensities = fl[tuple(indices)]

        if point_sampling < 1.0:
            n_points = indices.shape[1]
            keep = np.random.choice(n_points, int(n_points * point_sampling), replace=False)
            indices = indices[:, keep]
            intensities = intensities[keep]

        coords_xyz = np.diag([dx, dy, dz]) @ indices[[2, 1, 0], :]  # (3, N) in microns
        r_proj = self.quaternion_to_rotation_matrix(e)[:2, :]
        y_microns = s * (r_proj @ coords_xyz) + d  # shape: (2, N)
        y_pixels = y_microns / np.array([[dx_fib], [dy_fib]])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.fib_image, cmap='gray', origin='upper')

        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            X_scaled = self.scale_fl_voxels(X)
            Y_scaled = self.scale_fib_pixels(Y)
            Y_pred = self.transform_points_quaternion(e, s, d, X_scaled)

            Y_scaled_pix = Y_scaled[[0, 1]] / np.array([[dx_fib], [dy_fib]])
            Y_pred_pix = Y_pred[[0, 1]] / np.array([[dx_fib], [dy_fib]])

            ax.scatter(Y_scaled_pix[0], Y_scaled_pix[1], c='lime', s=50, marker='x', label='FIB Fiducials')
            ax.scatter(Y_pred_pix[0], Y_pred_pix[1], c='cyan', s=40, marker='o',
                       edgecolors='white', label='Projected FL Fiducials')

        ax.set_title("Overlay: FL Projection + Fiducials")
        ax.axis('off')

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='lower right')

        plt.savefig(os.path.join(self.path, 'Fiducial_registration_overlay_all.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_all_overlays(self):
        """
        Generates and saves:
        1. Thresholded FL point projection over FIB
        2. Z-weighted MIP overlay
        3. Flat-Z MIP overlay
        4. Multi-page TIFF with both MIPs
        """
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        intensity_threshold=0.5
        point_sampling=0.5
        e = self.e_final / np.linalg.norm(self.e_final)
        s = self.s_final
        d = self.d_final.reshape(2, 1)
        r_proj = self.quaternion_to_rotation_matrix(e)[:2, :]
        dz, dy, dx = self.fl_scale
        dy_fib, dx_fib = self.fib_scale[1:]
        fib_image = self.fib_image

        def project_coords(coords_3d):
            projected = s * (r_proj @ coords_3d) + d
            return np.vstack([
                projected[0, :] / dx_fib,
                projected[1, :] / dy_fib])

        def plot_overlay(save_path, proj_pix, title, intensities, use_alpha=False, alpha_max=0.8):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(fib_image, cmap='gray', origin='upper')

            if use_alpha:
                # Normalize the passed-in intensities
                norm = Normalize(vmin=np.min(intensities), vmax=np.max(intensities))
                intensities_norm = norm(intensities)

                # Get RGBA values from colormap
                cmap = cm.get_cmap('inferno')
                rgba = cmap(intensities_norm)

                # Apply linear alpha scaling
                rgba[:, 3] = intensities_norm * alpha_max

                ax.scatter(proj_pix[0], proj_pix[1], color=rgba, s=1)
            else:
                ax.scatter(proj_pix[0], proj_pix[1], c=intensities, cmap='inferno', s=1, alpha=0.5)

            ax.set_title(title)
            ax.axis('off')
            fig.savefig(os.path.join(self.path, save_path), dpi=300, bbox_inches='tight', transparent=use_alpha)
            plt.close(fig)

        fl_proj = self.fl_stack[0][0][:]  # shape (Z, Y, X)
        indices = np.array(np.nonzero(fl_proj > intensity_threshold))
        intensities_proj = fl_proj[tuple(indices)]

        if point_sampling < 1.0:
            n_points = int(indices.shape[1] * point_sampling)
            keep = np.random.choice(indices.shape[1], n_points, replace=False)
            indices = indices[:, keep]
            intensities_proj = intensities_proj[keep]

        coords_proj = np.diag([dx, dy, dz]) @ indices[[2, 1, 0], :]  # X, Y, Z → reorder
        proj_pix = project_coords(coords_proj)
        plot_overlay("FL_Stack_Projection_Overlay.png", proj_pix, "FL Projection Overlayed on FIB", intensities_proj)
        fl_mip = self.fl_stack[0][1]  # shape (Z, Y, X)
        Z, Y, X = fl_mip.shape
        mip = np.max(fl_mip, axis=0)
        z_indices = np.argmax(fl_mip, axis=0)
        intensities_mip = mip.ravel()
        y_coords, x_coords = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
        x_real = x_coords.ravel() * dx
        y_real = y_coords.ravel() * dy

        z_weighted = z_indices.ravel() * dz
        coords_zweighted = np.vstack([x_real, y_real, z_weighted])
        proj_zweighted_pix = project_coords(coords_zweighted)
        plot_overlay("mip_overlay_zweighted.png", proj_zweighted_pix, "Z-weighted MIP Overlay", intensities_mip, use_alpha=True)

        mid_z = (Z // 2) * dz
        z_flat = np.full_like(x_real, mid_z)
        coords_flat = np.vstack([x_real, y_real, z_flat])
        proj_flat_pix = project_coords(coords_flat)
        plot_overlay("mip_overlay_flat.png", proj_flat_pix, "Flat MIP Overlay", intensities_mip, use_alpha=True)

        img1 = Image.open(os.path.join(self.path, "mip_overlay_zweighted.png")).convert("RGB")
        img2 = Image.open(os.path.join(self.path, "mip_overlay_flat.png")).convert("RGB")
        img1.save(os.path.join(self.path, "mip_overlays.tiff"), save_all=True, append_images=[img2])

    def import_images(self, path):
        """
        Currently optimized for Meteor data import. Have to check if Thermo needs a different setup, or if we can
        agree on creating a consistent file format.
        """
        try:
            img = AICSImage(path)
            xml_str = img.metadata
            root = ET.fromstring(xml_str)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            pixels = root.find(".//ome:Pixels", ns)
        except:
            with TiffFile(path) as tif:
                img = tif.asarray()
                image_description = tif.pages[0].tags["ImageDescription"].value

                xml_str = image_description.decode("utf-8") if isinstance(image_description,
                                                                          bytes) else image_description
                root = ET.fromstring(xml_str)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                pixels = root.find(".//ome:Pixels", ns)

        scale = [pixels.attrib.get("PhysicalSizeZ"), pixels.attrib.get("PhysicalSizeY"), pixels.attrib.get("PhysicalSizeX")]
        scale_float = [float(x) if x is not None else 1.0 for x in scale]
        return img.data, tuple(scale_float)




fl_fiducials = [
    np.array([[257, 313, 338, 351], [489, 465, 432, 390], [10, 10, 10, 10]])
]


fib_fiducials = [
    np.array([[916, 1053, 1122, 1160], [558, 553, 535, 504]])
]



path = '/Users/sophia.betzler/Desktop/3DCT_Test/'
fib_image_path = 'Feature-5-post-rough-mill-FIB-005.ome.tiff'
fl_stack_path = 'multi-channel-z-stack-post-rough-mill-Feature-5-Ready to Mill-005.ome.tiff'
transform = Transformation(fl_fiducials=fl_fiducials, fib_fiducials=fib_fiducials,
                           fib_image_name=fib_image_path, fl_stack_name=fl_stack_path,
                           path=path)
params_init = transform.guess_parameters_svd()
transform.run_transformation(params_init)

##### Export transformed FL Stack
##### WRITE FUNCTION TO PERFORM A GAUSSIAN FIT AND CHECK IF DENOISING WORKS FOR FL IMAGES
##### RESERACH METHODS TO COMPARE PICKS AND REMOVE PICKS WHICH WERE ONLY FOUND IN ONE INSTANCE
##### SEGMENT ANYTHING DOESN"T WORK VERY WELL IN THE FIRST TEST
##### NEED TO IMPROVE VISIBILITY OF SPOTBURNS, PROBABLY NEED TO CUT OFF OUTSIDE SLICES, HOW CAN I DETERMINE RANGE IN STACK





