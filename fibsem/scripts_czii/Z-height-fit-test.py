
from aicsimageio import AICSImage
import tifffile
import xml.etree.ElementTree as ET
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
from sympy.codegen.ast import continue_


def import_images(path, fl=False):
    img = AICSImage(path)
    if fl:
        image = img.data[0][1]
    else:
        image = img.data[0][0]
    try:
        xml_str = img.metadata
        root = ET.fromstring(xml_str)
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        pixels = root.find(".//ome:Pixels", ns)
        pixel_z = float(pixels.attrib.get("PhysicalSizeZ"))
        pixel_y = float(pixels.attrib.get("PhysicalSizeY"))
        pixel_x = float(pixels.attrib.get("PhysicalSizeX"))
    except:
        ome = img.metadata  # already parsed OME object
        pixel_z = float(ome.images[0].pixels.physical_size_z)
        pixel_y = float(ome.images[0].pixels.physical_size_y)
        pixel_x = float(ome.images[0].pixels.physical_size_x)
    return image, (pixel_z, pixel_y, pixel_x)

class ZHeightDetermination():
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
                    print(f"Fit failed for degree {degree}: {e}")
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

    def determine_2d_spotburn_center(self, x0, y0, z_height, path, window_size=14, use_aic=True, max_degree=3):
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
                    loss='soft_l1'
                )

                fit_surface = gaussian_2d_with_poly(result.x, x_grid, y_grid, degree, array_interp.shape)
                mse = mean_squared_error(array_interp.ravel(), fit_surface.ravel())
                score = array_interp.size * np.log(mse) + 2 * len(result.x) if use_aic else mse

                if score < best_score:
                    best_score = score
                    best_result = result
                    best_degree = degree
                    best_fit_surface = fit_surface

            except Exception as e:
                print(f"Degree {degree} fit failed: {e}")

        if best_result is None:
            raise RuntimeError("All polynomial background fits failed.")

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

        def remove_duplicates(result, threshold=1.0):
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
                        if np.all(np.abs(result[i] - result[j]) < threshold):
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
            for z in range(len(image)):
                intensity.append(image[z, int(spotburn[1]), int(spotburn[2])])

            z_height, r2_score_value = self.determine_z_height(x0=int(spotburn[2]), y0=int(spotburn[1]), window_size=7,
                                                               path=spotburn_path, z_height_init=int(spotburn[0]))
            print(f"The determined Z-height is {z_height} with a score of {r2_score_value}.")
            if r2_score_value > 0.70:
                fitted_center, minimum_center, score = (self.determine_2d_spotburn_center(x0=int(spotburn[2]),
                                                      y0=int(spotburn[1]),
                                                      z_height=z_height,
                                                      path=spotburn_path))
                x0, y0 = minimum_center
                print(f"The minimum center is {int(x0), int(y0)}, the original values are {int(spotburn[1]), int(spotburn[2])}")
                print(f"The score of the 2D Gaussian Fit is {score}.")

                z_height_fine, _ = self.determine_z_height(x0= int(x0), y0=int(y0),
                                                           z_height_init=int(z_height),
                                                           window_size=5,
                                                           path=spotburn_path)
                if abs(z_height_fine - spotburn[0]) < 2.0:
                    list_of_determined_z_heights.append((z_height_fine, spotburn[1], spotburn[2]))
                else:
                    print(f"The determine Z-height is very different from the one identified with ML. spotburn skipped.")

            else:
                print(
                    f"Z-height fit wasn't precise enough. To avoid wrong correlation results this spotburn was skipped."
                    f"The r2 score was {r2_score_value}.")

        list_of_spotburns = sorted(list_of_determined_z_heights, key=lambda v: (v[2], v[1], v[0]))
        remove_duplicates(list_of_spotburns)

        y_vals = [p[1] for p in list_of_determined_z_heights]
        x_vals = [p[2] for p in list_of_determined_z_heights]

        max_proj = np.max(image, axis=0)

        plt.figure(figsize=(12, 5))
        plt.imshow(max_proj, cmap='gray')
        plt.scatter(x_vals, y_vals, marker='x', color='w', label='optimized position')
        plt.scatter(spotburn[2], spotburn[1], marker='o', color='r', label='determined by ML')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + 'MIP_with_spotburns.png', dpi=300, bbox_inches='tight')
        plt.close()
        return list_of_spotburns



list_of_spotburns = np.load('/Users/sophia.betzler/Desktop/detected_spotburns.npy')
path = '/Users/sophia.betzler/Documents/Data/Hydra/20250327-delmic-patrick/'
image, scale = import_images(path + 'image-Feature-1-Polished-001.ome.tiff')
z_height_det = ZHeightDetermination(fl_stack=image, list_of_spotburns=list_of_spotburns, path=path)
final_spotburn_location = z_height_det.run_fitting()




### CHECK IF IT WORKS BETTER IF YOU USE AN ARIA INSTEAD OF A PIXEL?
### CHECK MULTIPLE IMAGES TO SEE HOW MANY I CAN FIT WELL
### --> WE WOULD PROBABLY NEED ABOUT 5 PER IMAGE