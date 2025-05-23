import os
import json
import time
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.interpolate import RectBivariateSpline, interp1d
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN
import itertools
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
from typing import Any
import copy
from scipy.spatial.transform import Rotation as R



########################################################################################################################
#
#       Function which is runs the full automated CLEM Pipeline
#
########################################################################################################################
class AutomaticCLEM:
    def __init__(self, bf, target_position, lamella_top_y):
        self.bf = bf
        self.folder_path = self.bf.folder_path
        self.temp_folder_path = self.bf.temp_folder_path
        self.target_position = target_position
        self.lamella_top_y = lamella_top_y

    def run_full_3dct_pipeline(self):
        self.identify_spotburns_fl_stack()
        print(f"[INFO] Spotburn identification in FL stack DONE. {len(self.ml_spotburn_location_fl)} "
              f"Spotburns found.")
        self.identify_spotburns_fib_image()
        print(f"[INFO] Spotburns identification in FIB image DONE. {len(self.ml_spotburn_location_fib)} "
              f"Spotburns found.")
        self.verify_identified_fiducials()
        print(f"[INFO] The final Spotburns are {self.fib_spotburns_final}  /n and {self.fl_spotburns_final}.")
        self.correlation_3dct()
        print(f"[INFO] 3DCT DONE.")


    def identify_spotburns_fl_stack(self):
        self.fl_stack, self.fl_scale = self.bf.import_images(self.folder_path + 'FL_Z_stack.tiff')
        self.fl_stack_fl, _ = self.bf.import_images(self.folder_path + 'FL_Z_stack.tiff', fl=True)
        np.save(os.path.join(self.temp_folder_path, "FL_Z_Stack.npy"), self.fl_stack)

        self.bf.execute_external_script(script='Identify_Spotburns_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=[self.folder_path, 'FL'])
        while not os.path.exists(os.path.join(self.temp_folder_path, 'spotburns_id_result.json')):
            time.sleep(0.1)

        with open(os.path.join(self.temp_folder_path, 'spotburns_id_result.json'), 'r') as file:
            list_spotburns_fl_ml = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'spotburns_id_result.json'))

        z_height_determination = ZHeightDetermination(fl_stack=self.fl_stack,
                                                      list_of_spotburns=list_spotburns_fl_ml,
                                                      path=self.folder_path)

        self.ml_spotburn_location_fl = z_height_determination.run_fitting()
        print(f"The spotburn location in fl are {self.ml_spotburn_location_fl}.")

    def identify_spotburns_fib_image(self):

        self.fib_image, self.fib_scale = self.bf.import_images(self.folder_path + 'FIB_image.tiff')
        print(self.fib_scale)

        np.save(os.path.join(self.temp_folder_path, "FIB_Image.npy"), self.fib_image)

        self.bf.execute_external_script(script='Identify_Spotburns_Remote.py',
                                        dir_name='Ultralytics',
                                        parameter=[self.folder_path, 'FIB'])

        while not os.path.exists(os.path.join(self.temp_folder_path, 'spotburns_id_result.json')):
            time.sleep(0.1)

        with open(os.path.join(self.temp_folder_path, 'spotburns_id_result.json'), 'r') as file:
            list_spotburns_fib_ml = json.load(file)

        os.remove(os.path.join(self.temp_folder_path, 'spotburns_id_result.json'))

        spotburn_location_fib = sorted(list_spotburns_fib_ml, key=lambda v: (v[2], v[1], v[0]))
        self.ml_spotburn_location_fib = [entry[1:] for entry in spotburn_location_fib]
        print(f"The spotburns identified in fib are {self.ml_spotburn_location_fib}")

    def verify_identified_fiducials(self):


        spotburn_matching = SpotburnMatching(fib_spotburns=self.ml_spotburn_location_fib,
                                         fl_spotburns=self.ml_spotburn_location_fl,
                                         fib_scale=self.fib_scale,
                                         fl_scale=self.fl_scale,
                                         path=self.folder_path,
                                         method='thorough')

        self.fl_spotburns_final, self.fib_spotburns_final= spotburn_matching.run_spotburn_matching()
        print(f"The final position of the spotburns in the fl are {self.fl_spotburns_final}. The "
              f"final position of the spotburns in fib are {self.fib_spotburns_final}")

    def correlation_3dct(self):

        transformation_3d = Transformation3DCT(fl_fiducials=self.fl_spotburns_final,
                                               fib_fiducials=self.fib_spotburns_final,
                                               fib_image=self.fib_image, fib_scale=self.fib_scale,
                                               fl_stack=self.fl_stack,
                                               fl_stack_fl=self.fl_stack_fl,
                                               fl_scale=self.fl_scale,
                                               path=self.folder_path,
                                               target_FL_position=self.target_position, #[z, y, x] in pixel
                                               lamella_top_y=self.lamella_top_y) #in pixel

        params_init = transformation_3d.guess_parameters_svd()
        transformation_3d.run_transformation(params_init)

########################################################################################################################
#
#       Class which does the Fiducial Verification and Fine-Z-height Determination for the FL-Stack
#
########################################################################################################################

class ZHeightDetermination:
    def __init__(self, fl_stack, list_of_spotburns, path):
        self.list_of_spotburn = list_of_spotburns
        self.fl_stack = fl_stack
        self.path = path

    def determine_z_height(self, x0, y0, path, window_size, z_height_init):

        def gaussian_on_poly_bg(x, A, mu, sigma, *poly_coeffs):
            """Gaussian dip + polynomial background."""
            poly = np.polyval(poly_coeffs, x)
            gauss = -A * np.exp(-((x - mu)**2) / (2 * sigma**2))
            return gauss + poly

        def double_gaussian_with_poly(x, A1, mu1, sigma1, A2, mu2, sigma2, *poly_coeffs):
            """
            Two Gaussians (positive + negative) + polynomial background.
            - A1, mu1, sigma1: broad positive Gaussian
            - A2, mu2, sigma2: narrow negative Gaussian
            - poly_coeffs: background polynomial coefficients (highest degree first)
            """
            G1 = A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
            G2 = -A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
            background = np.polyval(poly_coeffs, x)
            return G1 + G2 + background

        def residuals(params, x, y, poly_degree):
            A, mu, sigma = params[:3]
            poly_coeffs = params[3:3 + poly_degree + 1]
            y_model = gaussian_on_poly_bg(x, A, mu, sigma, *poly_coeffs)
            return y_model - y

        def interpolation_1d(x, y):
            """
            Interpolates the provided line_plots to make the fit more robust.
            """
            f_interp = interp1d(x, y, kind='cubic')
            x_interp = np.linspace(x.min(), x.max(), len(x)*4)  # new shape: (105,)
            y_interp = f_interp(x_interp)
            return x_interp, y_interp

        def guess_dog(x, y, degree):
            A1_init = (np.max(y) - np.min(y)) / 2
            mu1_init = x[np.argmax(y)]
            sigma1_init = (x[-1] - x[0]) / 4
            A2_init = np.max(y) - np.min(y)
            mu2_init = x[np.argmin(y)]
            sigma2_init = sigma1_init / 2
            poly_init = [0] * degree + [np.median(y)]
            return ([A1_init, mu1_init, sigma1_init, A2_init, mu2_init, sigma2_init] + poly_init)

        def bounds_dog(x, degree):
            lower_bounds = [0, x.min(), 0.1, 0, x.min(), 0.1] + [-np.inf] * (degree + 1)
            upper_bounds = [np.inf, x.max(), (x.max() - x.min()), np.inf, x.max(), (x.max() - x.min())] + [np.inf] * (degree + 1)
            return lower_bounds, upper_bounds

        def guess_neg_gauss(x, y, degree):
            A = np.abs(np.max(y) - np.min(y))
            mu = x[np.argmin(y)]
            sigma = (x[-1] - x[0]) / 6
            poly = list(np.polyfit(x, y, degree))
            return [A, mu, sigma] + poly

        def bounds_neg_gauss(x, degree):
            return ([0, x.min(), 0.1] + [-np.inf] * (degree + 1),
                [np.inf, x.max(), 4.0] + [np.inf] * (degree + 1))

        def crop_z_window(z_center, half_window=window_size, z_max=self.fl_stack.shape[0]):
            """
            Safely crops a Z window around a center slice, ensuring it stays within bounds.
            """
            start = max(0, z_center - half_window)
            end = min(z_max, z_center + half_window + 1)  # +1 because slicing is exclusive
            return int(start), int(end)

        def fit_model(x, y, model_func, guess_func, bounds_func, max_degree=5, robust=True, loss_func='soft_l1',
                      use_aic=True):
            best_fit = {'score': np.inf}

            for degree in range(max_degree + 1):
                init_params = guess_func(x, y, degree)
                lower, upper = bounds_func(x, degree)

                try:
                    if robust:
                        res = least_squares(
                            residuals, x0=init_params, bounds=(lower, upper),
                            args=(x, y, model_func, degree), loss=loss_func
                        )
                        params = res.x
                    else:
                        from scipy.optimize import curve_fit
                        params, _ = curve_fit(lambda x, *p: model_func(x, *p),
                                    x, y, p0=init_params, bounds=(lower, upper))

                    y_fit = model_func(x, *params)
                    mse = mean_squared_error(y, y_fit)
                    score = len(x) * np.log(mse) + 2 * len(params) if use_aic else mse

                    if score < best_fit['score']:
                        best_fit.update({
                            'best_order': degree,
                            'params': params,
                            'y_fit': y_fit,
                            'score': score,
                            'r2': r2_score(y, y_fit)})
                except Exception as e:
                    print(f" [DEGUB] Fit failed for degree {degree}: {e}")
            return best_fit

        def plot_z_fit_result(x, y, x_interp, y_interp, fit_result, start, path, window_size,
                              label="Determine_Z_height"):
            """
            Plots the raw and fitted Z-profile data, saves the figure, and computes final z-height.
            """

            plt.figure(figsize=(6, 4))
            plt.plot(x, y, label="Raw data", marker='o', linestyle='--', alpha=0.6)
            plt.plot(x_interp, y_interp, label="Interpolated", alpha=0.6)
            plt.plot(x_interp, fit_result['y_fit'], label="Best fit", linewidth=2)
            plt.title("Z-profile Gaussian Fit")
            plt.xlabel("Z index (relative)")
            plt.ylabel("Intensity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fname=f"{path}{label}_{window_size}.png", dpi=300, bbox_inches='tight')
            plt.close()
            z_height = fit_result['params'][1]
            return z_height, fit_result['r2']

        start, end = crop_z_window(int(z_height_init))
        y = self.fl_stack[start:end, int(y0), int(x0)]
        x = np.arange(len(y))+start
        x_interp, y_interp = interpolation_1d(x, y)

        result = fit_model(x_interp, y_interp,
                        model_func=gaussian_on_poly_bg,
                        guess_func=guess_neg_gauss,
                        bounds_func=bounds_neg_gauss,
                        max_degree=5,
                        robust=False)

        z_height, r2 = plot_z_fit_result(x, y, x_interp, y_interp, result, start, path, window_size)
        return z_height, r2

    def determine_2d_spotburn_center(self, x0, y0, z_height, path, window_size=20, use_aic=True, max_degree=3):
        """
        This function finds the center of the spot-burn based on the previously identified Z-height-slice. The
        window_size is important to take the spot-burn size into account. Should be constant if the same experimental
        conditions are used.
        """
        def build_poly_basis(x, y, degree):
            return np.array([(x**i) * (y**j) for i in range(degree + 1) for j in range(degree + 1 - i)])

        def gaussian_2d_with_poly(params, x, y, degree, image_shape):
            A, x0, y0, sigma_x, sigma_y, theta = params[:6]
            offset = params[6]
            poly_params = params[7:]

            x_flat, y_flat = x.ravel(), y.ravel()

            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            gauss = -A * np.exp(
                -(a * (x_flat - x0) ** 2 + 2 * b * (x_flat - x0) * (y_flat - y0) + c * (y_flat - y0) ** 2))

            poly_bg = np.dot(poly_params, build_poly_basis(x_flat, y_flat, degree))
            model_flat = offset + gauss + poly_bg
            return (offset + gauss + poly_bg).reshape(image_shape)

        def residuals(params, x, y, data, degree):
            return (gaussian_2d_with_poly(params, x, y, degree, data.shape) - data).ravel()

        def plot_spotburn_fit_result(image, fit, x0_fit, y0_fit, out_path, filename='Fit_spotburn_center.png'):
            min_idx = np.unravel_index(np.argmin(fit), fit.shape)
            x_min, y_min = min_idx[1], min_idx[0]

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("Interpolated Input")
            plt.imshow(array_interp, cmap='inferno')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Fitted 2D Gaussian Dip")
            plt.imshow(fit_surface, cmap='inferno')
            plt.scatter(x0_fit, y0_fit, s=12, marker='x', c='white', label='Fitted center')
            plt.scatter(x_min, y_min, s=12, marker='x', c='red', label='Minimum')
            plt.legend()
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(path + filename, dpi=300, bbox_inches='tight')
            plt.close()

        slice_idx = int(round(z_height))
        half_window = window_size // 2
        crop = self.fl_stack[slice_idx, y0 - half_window:y0 + half_window, x0 - half_window:x0 + half_window]

        interpolation_scale = 4
        y = np.arange(crop.shape[0])
        x = np.arange(crop.shape[1])
        interp_func = RectBivariateSpline(y, x, crop)
        y_interp = np.linspace(0, crop.shape[0] - 1, crop.shape[0] * interpolation_scale)
        x_interp = np.linspace(0, crop.shape[1] - 1, crop.shape[1] * interpolation_scale)
        array_interp = interp_func(y_interp, x_interp)

        H, W = array_interp.shape
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))

        best_result = None
        best_score = np.inf

        for degree in range(max_degree + 1):
            n_poly = len(build_poly_basis(x_grid, y_grid, degree))

            A_init = np.max(array_interp) - np.min(array_interp)
            x0_init = W / 2
            y0_init = H / 2
            sigma_init = W / 10
            theta_init = 0
            offset_init = np.median(array_interp)
            poly_init = [0] * n_poly

            initial_guess = [A_init, x0_init, y0_init, sigma_init, sigma_init, theta_init, offset_init] + poly_init
            lower_bounds = [0, 0, 0, 1, 1, -np.pi / 2, 0] + [-np.inf] * n_poly
            upper_bounds = [np.inf, W, H, W, H, np.pi / 2, np.max(array_interp)] + [np.inf] * n_poly

            try:
                result = least_squares(
                    residuals,
                    x0=initial_guess,
                    bounds=(lower_bounds, upper_bounds),
                    args=(x_grid, y_grid, array_interp, degree),
                    loss='soft_l1')

                fit_surface = gaussian_2d_with_poly(result.x, x_grid, y_grid, degree, array_interp.shape)
                mse = mean_squared_error(array_interp.ravel(), fit_surface.ravel())
                score = array_interp.size * np.log(mse) + 2 * len(result.x) if use_aic else mse

                if score < best_score:
                    best_score = score
                    best_result = result
                    best_degree = degree
                    best_fit_surface = fit_surface

            except Exception as e:
                print(f"[DEBUG] Degree {degree} fit failed: {e}")

        if best_result is None:
            raise RuntimeError("[DEBUG] All polynomial background fits failed.")

        params = best_result.x
        x0_fit, y0_fit = params[1], params[2]

        if 'best_fit_surface' in locals() and best_fit_surface is not None:
            min_idx = np.unravel_index(np.argmin(best_fit_surface), best_fit_surface.shape)
            x_min, y_min = min_idx[1], min_idx[0]
        else:
            x_min, y_min = 0, 0

        fitted_center = [int(np.round(x0 - half_window + x0_fit / interpolation_scale)),
                        int(np.round(y0 + half_window - y0_fit / interpolation_scale))]

        minimum_center = [int(np.round(x0 - half_window + x_min / interpolation_scale)),
                        int(np.round(y0 + half_window - y_min / interpolation_scale))]

        plot_spotburn_fit_result(array_interp, best_fit_surface, x0_fit, y0_fit, path)

        return fitted_center, minimum_center, best_score

    def run_fitting(self):
        """
        Function to run the fitting workflow. Three-step process.
        """

        def remove_duplicates(result, threshold=3.0):
            result = np.array(result)
            keep = []
            visited = np.zeros(len(result), dtype=bool)
            for i in range(len(result)):
                if visited[i]:
                    continue

                group = [result[i]]
                visited[i] = True

                for j in range(i + 1, len(result)):
                    if not visited[j]:
                        if np.all(np.abs(result[i][1:] - result[j][1:]) < threshold):
                            group.append(result[j])
                            visited[j] = True

                averaged = np.mean(group, axis=0)
                keep.append(averaged)

            return np.array(keep)

        list_of_determined_z_heights = []
        os.makedirs(os.path.join(self.path) + 'Fits_of_spotburns', exist_ok=True)

        for i, spotburn in enumerate(self.list_of_spotburn):

            spotburn_path = os.path.join(self.path) + 'Fits_of_spotburns/' + 'spotburn_' + str(i) + '/'
            os.makedirs(spotburn_path, exist_ok=True)

            intensity = []
            for z in range(len(self.fl_stack)):
                intensity.append(self.fl_stack[z, int(spotburn[1]), int(spotburn[2])])

            z_height, r2_score_value = self.determine_z_height(x0=int(spotburn[2]), y0=int(spotburn[1]), window_size=7,
                                                               path=spotburn_path, z_height_init=int(spotburn[0]))
            with open(os.path.join(self.path, "fit_log_z_spotburns.txt"), "a") as log:
                log.write(f"Spotburn {i}: Z-height is {z_height} with a score of {r2_score_value}.\n")
            if r2_score_value > 0.70:
                fitted_center, minimum_center, score = (self.determine_2d_spotburn_center(x0=int(spotburn[2]),
                                                      y0=int(spotburn[1]),
                                                      z_height=z_height,
                                                      path=spotburn_path))
                x0, y0 = minimum_center
                with open(os.path.join(self.path, "fit_log_z_spotburns.txt"), "a") as log:
                    log.write(f"Spotburn {i}: Fitted center minimum is {int(x0), int(y0)}, "
                              f"the original values are {int(spotburn[1]), int(spotburn[2])}"
                              f"with a score of {score}.\n")


                z_height_fine, _ = self.determine_z_height(x0= int(x0), y0=int(y0),
                                                           z_height_init=int(z_height),
                                                           window_size=5,
                                                           path=spotburn_path)
                if abs(z_height_fine - spotburn[0]) < 2.0:
                    list_of_determined_z_heights.append((z_height_fine, spotburn[1], spotburn[2]))
                    with open(os.path.join(self.path, "fit_log_z_spotburns.txt"), "a") as log:
                        log.write(f"Spotburn {i}: The final Z-height is {z_height_fine}.\n")
                else:
                    print(f"[DEBUG] The determine Z-height is very different from the one identified with ML. "
                          f"Spotburn skipped.")

            else:
                print(
                    f"[DEBUG] Z-height fit wasn't precise enough. To avoid wrong correlation results this spotburn was skipped."
                    f"[DEBUG] The r2 score was {r2_score_value}.")

        list_of_spotburns = sorted(list_of_determined_z_heights, key=lambda v: (v[2], v[1], v[0]))
        list_of_spotburns_fl = remove_duplicates(list_of_spotburns)

        y_vals = [p[1] for p in list_of_determined_z_heights]
        x_vals = [p[2] for p in list_of_determined_z_heights]

        max_proj = np.max(self.fl_stack, axis=0)

        plt.figure(figsize=(12, 5))
        plt.imshow(max_proj, cmap='gray')
        plt.scatter(x_vals, y_vals, marker='x', color='#90DEFF')
        plt.tight_layout()
        plt.savefig(self.path + 'MIP_with_spotburns.png', dpi=300, bbox_inches='tight')
        plt.close()
        return list_of_spotburns_fl

########################################################################################################################
#
#       Class which verifies the Sputburn Matching between FIB image and FL stack (projected to 2D)
#
########################################################################################################################


class SpotburnMatching:

    def __init__(self, fib_spotburns, fl_spotburns, fib_scale, fl_scale, path, method='thorough'):
        self.fl_spotburns_full_list = fl_spotburns
        self.fib_spotburns_full_list = fib_spotburns
        list_fl_spotburns = [entry[1:] for entry in fl_spotburns]
        self.list_fl_spotburns_restructured = [arr.tolist() for arr in list_fl_spotburns]
        self.list_fl_spotburns = [[x, y] for y, x in self.list_fl_spotburns_restructured]
        self.list_fib_spotburns = [[x, y] for y, x in fib_spotburns]
        self.list_fl_spotburns_with_z = [[x, y, z] for z, y, x in fl_spotburns]
        self.fib_scale = fib_scale[1:]
        self.fl_scale = fl_scale[1:]
        self.path = path
        self.method = method

    def run_spotburn_matching(self):
        if self.method == 'rough':
            final_fl_spotburns, final_fib_spotburns = self.rough_alignment()
        else:
            final_fl_spotburns, final_fib_spotburns = self.thorough_alignment()
        return final_fl_spotburns, final_fib_spotburns

    def rough_alignment(self):

        def build_affine_ransac_model(points_src, points_dst):
            """Build two RANSAC regressors: one for x, one for y"""
            model_x = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
            model_y = make_pipeline(PolynomialFeatures(1), RANSACRegressor())

            model_x.fit(points_src, points_dst[:, 0])
            model_y.fit(points_src, points_dst[:, 1])

            return model_x, model_y

        def align_points(points_src, model_x, model_y):
            """Apply fitted RANSAC regressors to align points"""
            x_aligned = model_x.predict(points_src)
            y_aligned = model_y.predict(points_src)
            return np.vstack([x_aligned, y_aligned]).T


        def estimate_similarity_transform(src, dst):
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

        def match_fiducials_with_alignment(distance_threshold=10.0, visualize=True):
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
            model_x, model_y = build_affine_ransac_model(fib_spotburns_scaled[:min_len],
                                                              fl_spotburns_scaled[:min_len])
            aligned_fib_spotburns = align_points(fib_spotburns_scaled, model_x, model_y)

            scale, angle_rad, translation = estimate_similarity_transform(fib_spotburns_scaled,
                                                                               aligned_fib_spotburns)
            angle_deg = np.rad2deg(angle_rad)

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

            if visualize:
                plt.figure(figsize=(8, 6))
                plt.title("Fiducial Matching: FL (blue) vs FIB View (red)")
                plt.scatter(fl_spotburns_scaled[:, 0], fl_spotburns_scaled[:, 1], label="FL - Stack", marker='X', color='#90DEFF', s=50)
                plt.scatter(aligned_fib_spotburns[:, 0], aligned_fib_spotburns[:, 1], label="FIB - Image", color='#AEF359', s=70)
                for i, j in matches:
                    p1 = fl_spotburns_scaled[i]
                    p2 = aligned_fib_spotburns[j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=0.7)
                plt.gca().invert_yaxis()
                plt.xlabel("X (µm)")
                plt.ylabel("Y (µm)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.path + 'Spotburn_Matching_Output_2.png'), dpi=300, bbox_inches='tight')


            matched_fl = fl_spotburns_scaled[[i for i, _ in matches]]
            matched_aligned_fib = aligned_fib_spotburns[[j for _, j in matches]]
            match_errors = np.linalg.norm(matched_fl - matched_aligned_fib, axis=1)
            mean_error = np.mean(match_errors)
            fl_spotburns_with_z = [arr.tolist() for arr in self.fl_spotburns_full_list]
            final_fl_spotburns = [fl_spotburns_with_z[i] for i, _ in matches]

            final_fib_spotburns = [self.fib_spotburns_full_list[j] for _, j in matches]
            print(f"[INFO] Final Transformation Parameters for spotburn correlation: \n"
                  f"Rotation {angle_deg:.2f}, Scale {scale:.5f}, "
                  f"Translation (µm): [{translation[0]:.2f}, {translation[1]:.2f}]")
            print(f"[INFO] Mean similarity error (Euclidean distance): {mean_error:.4f} µm and {len(matches)} matches found.")
            return final_fl_spotburns, final_fib_spotburns

        final_fl_spotburns, final_fib_spotburns = match_fiducials_with_alignment(distance_threshold=3.0, visualize=True)
        return final_fl_spotburns, final_fib_spotburns

    def thorough_alignment(self):
        # def match_points_fixed_scale_brute(
        #         src_pts, tgt_pts,
        #         scale=1.2, min_points=3,
        #         top_n=2500, shortest_as_source=False):
        #
        #     src_pts = np.asarray(src_pts, dtype=np.float64)
        #     tgt_pts = np.asarray(tgt_pts, dtype=np.float64)
        #
        #     D, Ns = src_pts.shape
        #     Nt = tgt_pts.shape[1]
        #     assert D == tgt_pts.shape[0]
        #
        #     self.was_swapped = False
        #     if shortest_as_source:
        #         if Ns > Nt:
        #             src_pts, tgt_pts = tgt_pts, src_pts
        #             Ns, Nt = Nt, Ns
        #             print("[INFO] Swapped: smaller set is now source.")
        #             self.was_swapped = True
        #     else:
        #         if Ns < Nt:
        #             src_pts, tgt_pts = tgt_pts, src_pts
        #             Ns, Nt = Nt, Ns
        #             print("[INFO] Swapped: larger set is now source.")
        #             self.was_swapped = True
        #
        #     candidate_models = []
        #
        #     combos = list(itertools.combinations(range(Ns), min_points))
        #     perms = list(itertools.permutations(range(Nt), min_points))
        #
        #     total = len(combos) * len(perms)
        #     print(f"[INFO] Total model hypotheses to test: {total}")
        #
        #     progress = 0
        #     for i_idx, i_combo in enumerate(combos):
        #         for j_idx, j_combo in enumerate(perms):
        #             progress += 1
        #             if progress % 1000 == 0 or progress == 1:
        #                 print(f"[STATUS] Processing model {progress}/{total} ({(progress / total) * 100:.1f}%)")
        #
        #             src_sel = src_pts[:, i_combo]
        #             tgt_sel = tgt_pts[:, j_combo]
        #
        #             Rmat, tvec = estimate_rigid_transform_fixed_scale(src_sel, tgt_sel, scale)
        #             src_all = Rmat @ (src_pts * scale) + tvec[:, None]
        #
        #             if Ns <= Nt:
        #                 permutations_to_use = itertools.permutations(range(Nt), Ns)
        #                 for full_j in permutations_to_use:
        #                     tgt_all = tgt_pts[:, list(full_j)]
        #                     pairing = list(zip(range(Ns), list(full_j)))
        #                     error = np.mean(np.linalg.norm(src_all - tgt_all, axis=0) ** 2)
        #                     candidate_models.append((error, Rmat, tvec, pairing))
        #             else:
        #                 permutations_to_use = itertools.permutations(range(Ns), Nt)
        #                 for full_i in permutations_to_use:
        #                     src_subset = src_all[:, list(full_i)]
        #                     tgt_all = tgt_pts
        #                     pairing = list(zip(list(full_i), range(Nt)))
        #                     error = np.mean(np.linalg.norm(src_subset - tgt_all, axis=0) ** 2)
        #                     candidate_models.append((error, Rmat, tvec, pairing))
        #
        #     print("[INFO] Sorting candidate models...")
        #     candidate_models.sort(key=lambda x: x[0])
        #     print("[INFO] Done.")
        #     return candidate_models[:top_n]
        import numpy as np
        import itertools
        from scipy.spatial.distance import cdist

        def match_points_fixed_scale_brute(
                src_pts, tgt_pts,
                scale=1.2, min_points=3,
                top_n=2500, shortest_as_source=False,
                tol=0.05, use_fallback=True):

            src_pts = np.asarray(src_pts, dtype=np.float64)
            tgt_pts = np.asarray(tgt_pts, dtype=np.float64)

            D, Ns = src_pts.shape
            Nt = tgt_pts.shape[1]
            assert D == tgt_pts.shape[0]

            self.was_swapped = False
            if shortest_as_source:
                if Ns > Nt:
                    src_pts, tgt_pts = tgt_pts, src_pts
                    Ns, Nt = Nt, Ns
                    print("[INFO] Swapped: smaller set is now source.")
                    self.was_swapped = True
            else:
                if Ns < Nt:
                    src_pts, tgt_pts = tgt_pts, src_pts
                    Ns, Nt = Nt, Ns
                    print("[INFO] Swapped: larger set is now source.")
                    self.was_swapped = True

            # --- Prefilter using median distance ratio ---
            D_src = cdist(src_pts.T, src_pts.T)
            D_tgt = cdist(tgt_pts.T, tgt_pts.T)

            scale_low, scale_high = scale * (1 - tol), scale * (1 + tol)

            valid_src_indices = set()
            valid_tgt_indices = set()

            for i in range(Ns):
                src_d = np.median(D_src[i][D_src[i] > 0])
                if src_d == 0:
                    continue
                for j in range(Nt):
                    tgt_d = np.median(D_tgt[j][D_tgt[j] > 0])
                    if tgt_d == 0:
                        continue
                    rel_error = abs(tgt_d - scale * src_d) / (scale * src_d + 1e-8)
                    if rel_error <= tol:
                        valid_src_indices.add(i)
                        valid_tgt_indices.add(j)

            valid_src_indices = sorted(valid_src_indices)
            valid_tgt_indices = sorted(valid_tgt_indices)

            if len(valid_src_indices) < min_points or len(valid_tgt_indices) < min_points:
                print("[WARN] Not enough valid points after filtering.")
                if use_fallback:
                    print("[INFO] Retrying with full point sets (no filtering)...")
                    # Use all original points
                    valid_src_indices = list(range(Ns))
                    valid_tgt_indices = list(range(Nt))
                else:
                    return []

            combos = list(itertools.combinations(valid_src_indices, min_points))
            perms = list(itertools.permutations(valid_tgt_indices, min_points))

            total = len(combos) * len(perms)
            print(f"[INFO] Total model hypotheses to test: {total}")

            candidate_models = []
            progress = 0

            for i_combo in combos:
                for j_combo in perms:
                    progress += 1
                    if progress % 1000 == 0 or progress == 1:
                        print(f"[STATUS] Processing model {progress}/{total} ({(progress / total) * 100:.1f}%)")

                    src_sel = src_pts[:, i_combo]
                    tgt_sel = tgt_pts[:, j_combo]

                    Rmat, tvec = estimate_rigid_transform_fixed_scale(src_sel, tgt_sel, scale)
                    src_all = Rmat @ (src_pts * scale) + tvec[:, None]

                    if Ns <= Nt:
                        dists = np.linalg.norm(src_all[:, :, None] - tgt_pts[:, None, :], axis=0)  # shape: (Ns, Nt)
                        indices = np.argmin(dists, axis=1)  # length Ns
                        pairing = list(zip(range(Ns), indices))
                        error = np.mean([dists[i, indices[i]] ** 2 for i in range(Ns)])
                    else:
                        dists = np.linalg.norm(tgt_pts[:, :, None] - src_all[:, None, :], axis=0)  # shape: (Nt, Ns)
                        indices = np.argmin(dists, axis=1)  # length Nt
                        pairing = list(zip(indices, range(Nt)))
                        error = np.mean([dists[j, indices[j]] ** 2 for j in range(Nt)])

                    candidate_models.append((error, Rmat, tvec, pairing))

            print("[INFO] Sorting candidate models...")
            candidate_models.sort(key=lambda x: x[0])
            print("[INFO] Done.")
            return candidate_models[:top_n]

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
            def correct_pairing(pairing):
                return [(j, i) for i, j in pairing] if self.was_swapped else pairing

            best_model = None
            best_score = np.inf
            best_metrics = {}

            for i, (error, R_seed, t_seed, pairing) in enumerate(models):
                print(f"[DEBUG] Evaluating model {i + 1}/{len(models)}")
                pairing = correct_pairing(pairing)
                idx_a, idx_b = zip(*pairing)
                paired_src = src_pts[:, list(idx_a)]
                paired_tgt = tgt_pts[:, list(idx_b)]

                R, t = R_seed, t_seed
                src_trans = R @ (src_pts * scale) + t[:, None]

                used_src = set(idx_a)
                used_tgt = set(idx_b)
                pairing_final = list(pairing)

                pairing = list(zip(idx_a, idx_b))
                residuals = [np.linalg.norm(src_trans[:, i] - tgt_pts[:, j]) for i, j in pairing]

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
                print(
                    f"[DEBUG] Model score: inliers={len(pairing_final)}, mean_residual={mean_residual:.3f}, rmse={rmse:.3f}")
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

                    matched_coordinates = [(src_pts[:, i], tgt_pts[:, j]) for i, j in pairing]

                    fig, ax = plt.subplots(figsize=(6, 6))
                    for src_pt, tgt_pt in matched_coordinates:
                        ax.plot([src_pt[0], tgt_pt[0]], [src_pt[1], tgt_pt[1]], 'k--', alpha=0.5)
                    ax.scatter([p[0] for p, _ in matched_coordinates], [p[1] for p, _ in matched_coordinates], c='red',
                               label='Matched Source')
                    ax.scatter([p[0] for _, p in matched_coordinates], [p[1] for _, p in matched_coordinates], c='blue',
                               label='Matched Target')
                    ax.set_title("Final Matched Pairs")
                    ax.legend()
                    ax.axis('equal')
                    plt.tight_layout()
                    plt.savefig(self.path + 'Spotburn_Matching_Pairs_Output.png', dpi=300, bbox_inches='tight')

                def extract_matched_coordinates_from_original(pairing, src_pts_original, tgt_pts_original):
                    """
                    Returns matched coordinates from the original input order (2D or 3D), regardless of swap.
                    """
                    src_array_trans = np.stack(src_pts_original, axis=0).T
                    tgt_array_trans = np.stack(tgt_pts_original, axis=0).T

                    src_coords = [src_array_trans[:, i] for i, _ in pairing]
                    tgt_coords = [tgt_array_trans[:, j] for _, j in pairing]


                    final_fib_spotburns = np.column_stack(src_coords)[[1, 0], :].T
                    final_fl_spotburns = np.column_stack(tgt_coords)[[2, 1, 0], :].T

                    return final_fl_spotburns, final_fib_spotburns

                final_fl_spotburns, final_fib_spotburns = extract_matched_coordinates_from_original(pairing=pairing,
                                                                                    src_pts_original=self.list_fib_spotburns,
                                                                                    tgt_pts_original=self.list_fl_spotburns_with_z)

                return best_model, best_metrics, final_fl_spotburns, final_fib_spotburns

        def run_full_fixed_scale_pipeline(src_pts, tgt_pts, candidate_models, scale=1.0, inlier_thresh=10.0, top_k=10,
                                          min_required_inliers=3):
            labels, _ = cluster_model_parameters(candidate_models)

            filtered_models = filter_models_by_cluster(candidate_models, labels, dominant_label=None, top_k=top_k)

            best_model, best_metrics, final_fl_spotburns, final_fib_spotburns = apply_ransac_and_greedy(
                src_pts, tgt_pts, filtered_models,
                scale=scale,
                inlier_thresh=inlier_thresh,
                min_required_inliers=min_required_inliers
            )

            if best_model:
                R, t, pairing, inliers = best_model
                src_trans = R @ (src_pts * scale) + t[:, None]

                fig, ax = plt.subplots(figsize=(6, 6))
                for idx, (i, j) in enumerate(pairing):
                    if inliers[idx]:
                        ax.plot([src_trans[0, i], tgt_pts[0, j]], [src_trans[1, i], tgt_pts[1, j]], 'k--', alpha=0.6)
                ax.scatter(src_trans[0], src_trans[1], marker='o', s=40, color='#008000', edgecolors='#000000', label='FIB Fiducials')
                ax.scatter(tgt_pts[0], tgt_pts[1], marker='X', s=70, color='#90DEFF', edgecolors='#000000', label='FL Fiducials')
                ax.set_title("Best Model Alignment (Top Cluster)")
                ax.axis('equal')
                ax.legend()
                plt.tight_layout()
                plt.savefig(self.path + 'Spotburn_Matching_Output.png', dpi=300, bbox_inches='tight')
                plt.close()

            return best_model, best_metrics, final_fl_spotburns, final_fib_spotburns

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

        def cluster_model_parameters(models, show_plot=False, plot_top_n_clusters=False, top_n=20):
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
                plt.scatter(angles, translations[:, 0], c=labels, cmap='tab20', s=10)
                plt.xlabel("Rotation Angle (rad)")
                plt.ylabel("Translation X")
                plt.title("All Model Clusters")
                plt.tight_layout()

            if plot_top_n_clusters:
                from collections import Counter
                label_counts = Counter(label for label in labels if label != -1)
                top_labels = [label for label, _ in label_counts.most_common(top_n)]
                mask = np.isin(labels, top_labels)

                plt.figure(figsize=(6, 5))
                plt.scatter(angles[mask], translations[:, 0][mask], c=labels[mask], cmap='tab20', s=10)
                plt.xlabel("Rotation Angle (rad)")
                plt.ylabel("Translation X")
                plt.title(f"Top {top_n} Clusters")
                plt.tight_layout()

            label_counts = dict(zip(unique_labels, counts))
            dominant_label = max((label for label in unique_labels if label != -1), key=lambda l: label_counts[l],
                                 default=None)
            print(f"[INFO] Dominant cluster: {dominant_label} ({label_counts.get(dominant_label, 0)} models)")

            return labels, dominant_label

        fl = np.array(self.list_fl_spotburns)
        fib = np.array(self.list_fib_spotburns)
        if len(fl) > len(fib):
            relative_scale = self.fl_scale[1]/self.fib_scale[1]
        else:
            relative_scale = self.fib_scale[1]/self.fl_scale[1]


        all_models = match_points_fixed_scale_brute(
            src_pts=fib.T,
            tgt_pts=fl.T,
            scale=relative_scale,
            min_points=3,
            shortest_as_source=False)

        best_model, metrics, final_fl_spotburns, final_fib_spotburns = run_full_fixed_scale_pipeline(
            src_pts=fib.T,
            tgt_pts=fl.T,
            candidate_models=all_models,
            scale=relative_scale,
            inlier_thresh=250.0
        )

        return final_fl_spotburns, final_fib_spotburns


########################################################################################################################
#
#       Class for the optimized 3DCT Transformation which also removes outliers
#
########################################################################################################################


class Transformation3DCT:

    def __init__(self, fib_fiducials, fl_fiducials, fl_stack, fl_stack_fl, fib_image,
                 fib_scale, fl_scale, path, target_FL_position, lamella_top_y):
        self.path = path
        self.fib_image = fib_image
        self.fl_stack = fl_stack
        self.fl_stack_fl = fl_stack_fl
        self.fl_scale = fl_scale
        self.fib_scale = fib_scale
        fib_fiducials = np.array(fib_fiducials).T
        fl_fiducials = np.array(fl_fiducials).T
        fl_fiducials = fl_fiducials[::-1]
        fib_fiducials = fib_fiducials[::-1]
        self.fib_fiducials = [fib_fiducials]
        self.fl_fiducials = [fl_fiducials]
        self.target_fl_position = target_FL_position # in pixels
        self.lamella_top_y = lamella_top_y

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
        print(f"The scale of the FL is {self.fl_scale}.")
        scale_matrix = np.diag([dx, dy, dz])
        print(f"The scale matrix is {scale_matrix}")
        print(f"The function returns {scale_matrix @ x}.")
        return scale_matrix @ x

    def scale_fib_pixels(self, y):
        """
        Scale 2D FIB coordinates by pixel size to get real-world micrometer units.
        """
        if self.fib_scale[2] < 1e-5:
            dy, dx = (np.array(self.fib_scale[1:]) * 1e6)
        else:
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

    # def estimate_similarity_transform_3d_to_2d(self, X, Y):
    #     """
    #     Estimate initial parameters (R, s, d) for 3D→2D projection using Procrustes (SVD) method.
    #     Inputs:
    #     - X: shape (3, N) → 3D source points (already scaled)
    #     - Y: shape (2, N) → 2D target points (already scaled)
    #     Returns:
    #     - q: quaternion representing rotation (len 4)
    #     - s: scale
    #     - d: translation (2D)
    #     """
    #     N = X.shape[1]
    #     mu_X = np.mean(X, axis=1, keepdims=True)
    #     mu_Y = np.mean(Y, axis=1, keepdims=True)
    #     Xc = X - mu_X
    #     Yc = Y - mu_Y
    #     A = Yc @ np.linalg.pinv(Xc)
    #     s = np.linalg.norm(A, ord='fro') / np.sqrt(2)
    #     R_2x3 = A / s
    #     u, _, vh = np.linalg.svd(R_2x3)
    #     R_full = np.eye(3)
    #     R_full[:2, :] = R_2x3
    #     rotation = R.from_matrix(R_full)
    #     q = rotation.as_quat()  # (x, y, z, w) format
    #     q = np.roll(q, 1)
    #     d = mu_Y - s * R_2x3 @ mu_X
    #     d = d.flatten()
    #     return q, s, d
    from scipy.spatial.transform import Rotation as R

    def estimate_similarity_transform_3d_to_2d(self, X, Y):
        """
        Estimate initial parameters (R, s, d) for 3D→2D projection properly.

        Inputs:
        - X: shape (3, N) → 3D source points (already scaled)
        - Y: shape (2, N) → 2D target points (already scaled)

        Returns:
        - q: quaternion representing rotation (len 4)
        - s: scale
        - d: translation (2D)
        """
        N = X.shape[1]

        # Center the points
        mu_X = np.mean(X, axis=1, keepdims=True)  # (3, 1)
        mu_Y = np.mean(Y, axis=1, keepdims=True)  # (2, 1)
        Xc = X - mu_X
        Yc = Y - mu_Y

        # Solve for linear transformation A
        A = Yc @ np.linalg.pinv(Xc)  # (2, 3)

        # Extract scale
        scale = np.linalg.norm(A, ord='fro') / np.sqrt(2)

        # Normalize A to get projection
        A_normalized = A / scale  # (2, 3)

        # Now, make a full 3x3 rotation matrix
        # First two rows are A_normalized
        R_est = np.zeros((3, 3))
        R_est[:2, :] = A_normalized

        # Set third row as the cross product to complete an orthonormal basis
        # Ensuring right-handedness
        R_est[2, :] = np.cross(R_est[0, :], R_est[1, :])

        # Re-orthogonalize using SVD (in case of slight numerical issues)
        U, _, Vt = np.linalg.svd(R_est)
        R_est = U @ Vt

        # Convert rotation matrix to quaternion
        rotation = R.from_matrix(R_est)
        q = rotation.as_quat()  # (x, y, z, w)
        q = np.roll(q, 1)  # Make it (w, x, y, z) to match your convention

        # Translation
        d = (mu_Y - scale * A_normalized @ mu_X).flatten()

        return q, scale, d

    def guess_parameters_svd(self):
        """
        Creates the initial parameters used to fit the function.
        """
        X_all = np.hstack([self.scale_fl_voxels(X) for X in self.fl_fiducials])
        Y_all = np.hstack([self.scale_fib_pixels(Y) for Y in self.fib_fiducials])
        q, s, d = self.estimate_similarity_transform_3d_to_2d(X_all, Y_all)
        print(f"The initial guess parameters are {q} and {s} and {d}")
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

    def check_point_scaling(self):
        """
        Quick sanity check: compare average point magnitudes
        after voxel/pixel scaling. Warn if mismatched.
        """
        X_all = np.hstack([self.scale_fl_voxels(X) for X in self.fl_fiducials])
        Y_all = np.hstack([self.scale_fib_pixels(Y) for Y in self.fib_fiducials])

        avg_norm_X = np.mean(np.linalg.norm(X_all, axis=0))
        avg_norm_Y = np.mean(np.linalg.norm(Y_all, axis=0))

        ratio = avg_norm_X / avg_norm_Y if avg_norm_Y != 0 else np.inf

        print("\n--- Sanity Check: Point Magnitude Comparison ---")
        print(f"Average 3D FL point magnitude (µm): {avg_norm_X:.4f}")
        print(f"Average 2D FIB point magnitude (µm): {avg_norm_Y:.4f}")
        print(f"Ratio (FL / FIB): {ratio:.4f}")

        if ratio > 10 or ratio < 0.1:
            print("⚠️ WARNING: Points are very differently scaled. Registration might fail!")
        else:
            print("✅ Point scales are reasonably matched.")

    def project_FL_point_to_FIB(self):
        """
        Projects a 3D FL target point (in microns) into FIB 2D pixel coordinates.

        Input:
        - target_microns: array-like, shape (3,) [x, y, z] in microns

        Output:
        - projected_pixel: array, shape (2,) [x_pix, y_pix]
        """
        # Apply rotation, scale, translation
        e = self.e_final / np.linalg.norm(self.e_final)
        r_full = self.quaternion_to_rotation_matrix(e)
        r_proj = r_full[:2, :]
        s = self.s_final
        d = self.d_final.reshape(2, 1)
        dz, dy, dx = self.fl_scale  # voxel sizes in microns
        x_vox, y_vox, z_vox = self.target_fl_position[2], self.target_fl_position[1], self.target_fl_position[0]

        # Scale voxel indices to real-world microns
        target_microns = np.array([x_vox * dx, y_vox * dy, z_vox * dz]).reshape(3, 1)
        target_2d_microns = s * (r_proj @ target_microns) + d  # shape (2,1)

        # Convert microns to pixels
        if self.fib_scale[2] < 1e-5:
            dy_fib, dx_fib = np.array(self.fib_scale[1:]) * 1e6  # microns per pixel
        else:
            dy_fib, dx_fib = self.fib_scale[1:]
        projected_target_pixels = (target_2d_microns / np.array([[dx_fib], [dy_fib]])).flatten()
        focused_shifted_pixels = self.lamella_top_y + 1.4*(projected_target_pixels[1] - self.lamella_top_y)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.fib_image[0], cmap='gray', origin='upper')

        # Plot the projected point
        ax.scatter(projected_target_pixels[0], projected_target_pixels[1], marker='X', s=70, c='red', edgecolors='black')

        ax.set_title("Projected Target Point on FIB Image")
        ax.axis('off')
        plt.savefig(os.path.join(self.path, 'Target_Position_FIB.png'), dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.fib_image[0][300:1000, 300:1000], cmap='gray', origin='upper')

        # Plot the projected point
        ax.scatter(projected_target_pixels[0]-300, projected_target_pixels[1]-300, edgecolors='k', marker='X', s=50, color='red')
        ax.scatter(projected_target_pixels[0]-300, focused_shifted_pixels-300, edgecolors='k', marker='X', s=50, color='orange')

        ax.set_title("Projected Target Point on FIB Image")
        ax.axis('off')
        plt.savefig(os.path.join(self.path, 'Target_Position_FIB_Focus_Shifted.png'), dpi=300, bbox_inches='tight')

    def run_transformation(self, params_init):
        """
        Function runs the two-step optimization process.
        """
        self.check_point_scaling()
        constraints = {'type': 'eq', 'fun': self.quaternion_constraint}
        result_robust = minimize(
            fun=self.objective_with_huber,
            x0=params_init,
            method='SLSQP',
            constraints=constraints,
            bounds=[(-1, 1)] * 4 + [(-2.0, 2.0)] + [(-np.inf, np.inf), (-np.inf, np.inf)],
            options={'disp': True, 'maxiter': 100}
        )
        params_first = result_robust.x
        print(f"--- After first (robust) optimization ---")
        print(f"Estimated Scale: {params_first[4]:.6f}")
        e_first, s_first, d_first = params_first[0:4], params_first[4], params_first[5:7]
        fl_inliers, fib_inliers = [], []
        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            y_pred = self.transform_points_quaternion(e_first, s_first, d_first, X)
            _, mask = self.huber_loss_with_mask(Y, y_pred)
            fl_inliers.append(X[:, mask])
            fib_inliers.append(Y[:, mask])
        loss_history = []

        def loss_callback(params):
            loss = self.least_squares_objective(params)
            loss_history.append(loss)

        result_refined = minimize(
            fun=self.least_squares_objective,
            x0=params_first,
            method='SLSQP',
            constraints=constraints,
            options={'disp': True, 'maxiter': 100},
            callback=loss_callback
        )

        params_final = result_refined.x
        self.e_final, self.s_final, self.d_final = params_final[0:4], params_final[4], params_final[5:7]

        print("\n--- Final Optimized Parameters (2nd Pass) ---")
        print("Quaternion:", self.e_final)
        print("Scale:", self.s_final)
        print("Translation:", self.d_final)

        # --- Report optimization history ---
        print("\n--- Optimization Summary ---")
        print(f"Total iterations: {result_refined.nit}")
        if len(loss_history) >= 2:
            print(f"Initial loss: {loss_history[0]:.4f}")
            print(f"Final loss: {loss_history[-1]:.4f}")
            print(f"Loss improvement: {loss_history[0] - loss_history[-1]:.4f}")
        else:
            print("Loss history contains less than 2 points.")

        self.generate_all_overlays()
        self.plot_fiducial_alignment()
        self.project_FL_point_to_FIB()

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
        ax.imshow(self.fib_image[0], cmap='gray')

        e = self.e_final / np.linalg.norm(self.e_final)
        s = self.s_final
        d = self.d_final.reshape(2, 1)
        if self.fib_scale[2] < 1e-5:
            dy_fib, dx_fib = np.array(self.fib_scale[1:]) * 1e6
        else:
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
        fl = self.fl_stack
        indices = np.array(np.nonzero(fl > intensity_threshold))
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
        ax.imshow(self.fib_image[0], cmap='gray', origin='upper')

        for X, Y in zip(self.fl_fiducials, self.fib_fiducials):
            X_scaled = self.scale_fl_voxels(X)
            Y_scaled = self.scale_fib_pixels(Y)
            Y_pred = self.transform_points_quaternion(e, s, d, X_scaled)

            Y_scaled_pix = Y_scaled[[0, 1]] / np.array([[dx_fib], [dy_fib]])
            Y_pred_pix = Y_pred[[0, 1]] / np.array([[dx_fib], [dy_fib]])

            ax.scatter(Y_scaled_pix[0], Y_scaled_pix[1], color='#008000', edgecolors='k', s=90, marker='o', label='FIB Fiducials')
            ax.scatter(Y_pred_pix[0], Y_pred_pix[1], color='#90DEFF', s=70, marker='X',
                       edgecolors='k', label='Projected FL Fiducials')

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
        if self.fib_scale[2] < 1e-5:
            dy_fib, dx_fib = (np.array(self.fib_scale[1:]) * 1e6)
        else:
            dy_fib, dx_fib = self.fib_scale[1:]
        fib_image = self.fib_image[0]

        def project_coords(coords_3d):
            projected = s * (r_proj @ coords_3d) + d
            return np.vstack([
                projected[0, :] / dx_fib,
                projected[1, :] / dy_fib])

        def plot_overlay(save_path, proj_pix, title, intensities, use_alpha=False, alpha_max=0.8, gamma=1.0):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(fib_image, cmap='gray', origin='upper')

            if use_alpha:
                # Normalize intensities
                norm = Normalize(vmin=np.min(intensities), vmax=np.max(intensities))
                intensities_norm = norm(intensities)

                # Apply gamma correction
                intensities_gamma = intensities_norm ** gamma

                # Colormap with gamma-adjusted intensity
                cmap = cm.get_cmap('hot')
                rgba = cmap(intensities_norm)  # Use original for color mapping

                # Modify alpha based on gamma-corrected intensities
                rgba[:, 3] = intensities_gamma * alpha_max

                ax.scatter(proj_pix[0], proj_pix[1], color=rgba, s=1)
            else:
                ax.scatter(proj_pix[0], proj_pix[1], c=intensities, cmap='hot', s=1, alpha=0.8)

            ax.set_title(title)
            ax.axis('off')
            fig.savefig(os.path.join(self.path, save_path), dpi=300, bbox_inches='tight', transparent=use_alpha)
            plt.close(fig)

        fl_proj = self.fl_stack_fl[:]
        # shape (Z, Y, X)
        indices = np.array(np.nonzero(fl_proj > intensity_threshold))
        intensities_proj = fl_proj[tuple(indices)]

        if point_sampling < 1.0:
            n_points = int(indices.shape[1] * point_sampling)
            keep = np.random.choice(indices.shape[1], n_points, replace=False)
            indices = indices[:, keep]
            intensities_proj = intensities_proj[keep]

        coords_proj = np.diag([dx, dy, dz]) @ indices[[2, 1, 0], :]  # X, Y, Z → reorder
        proj_pix = project_coords(coords_proj)
        plot_overlay("FL_Stack_Projection_Overlay.png", proj_pix, "FL Projection Overlayed on FIB", intensities_proj, use_alpha=True, gamma=1.5)
        fl_mip = self.fl_stack_fl  # shape (Z, Y, X)
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
        plot_overlay("mip_overlay_zweighted.png", proj_zweighted_pix, "Z-weighted MIP Overlay", intensities_mip, use_alpha=True, gamma=1.5)

        mid_z = (Z // 2) * dz
        z_flat = np.full_like(x_real, mid_z)
        coords_flat = np.vstack([x_real, y_real, z_flat])
        proj_flat_pix = project_coords(coords_flat)
        plot_overlay("mip_overlay_flat.png", proj_flat_pix, "Flat MIP Overlay", intensities_mip*0.5, use_alpha=True, gamma=3.0)

        img1 = Image.open(os.path.join(self.path, "mip_overlay_zweighted.png")).convert("RGB")
        img2 = Image.open(os.path.join(self.path, "mip_overlay_flat.png")).convert("RGB")
        img1.save(os.path.join(self.path, "mip_overlays.tiff"), save_all=True, append_images=[img2])

