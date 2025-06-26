import os
import re
import numpy as np
from PIL import Image
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tifffile

def import_images_from_folder(folder_path):
    pattern = re.compile(r'image_(\d+)\.tif$')
    image_files = sorted(
        [f for f in os.listdir(folder_path) if pattern.match(f)],
        key=lambda x: float(pattern.match(x).group(1))
    )

    image_stack = []
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = tifffile.imread(img_path)  # preserves 16-bit
        image_stack.append(img)

    stack = np.stack(image_stack, axis=0)
    return stack


class FitFunctionsForStack:
    def __init__(self, image_stack):
        self.stack = image_stack

    def fit_gaussian_model(self, model_func, degree=1):

        def build_poly_basis(x, y, degree):
            return np.array([(x ** i) * (y ** j) for i in range(degree + 1) for j in range(degree + 1 - i)])

        def symmetric_2d_gaussian(coords, A, x0, y0, sigma, offset):
            x, y = coords
            return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) + offset

        def asymmetric_2d_gaussian(coords, A, x0, y0, sigma_x, sigma_y, theta, offset):
            x, y = coords
            x0 = float(x0)
            y0 = float(y0)
            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = (-np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            return A * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)) + offset

        def symmetric_gaussian_2d_with_poly_bg(params, x, y, degree, image_shape):
            A, x0, y0, sigma = params[:4]
            offset = params[4]
            poly_params = params[5:]
            x_flat, y_flat = x.ravel(), y.ravel()
            r2 = (x_flat - x0) ** 2 + (y_flat - y0) ** 2
            gauss = -A * np.exp(-r2 / (2 * sigma ** 2))

            poly_bg = np.dot(poly_params, build_poly_basis(x_flat, y_flat, degree))
            model_flat = offset + gauss + poly_bg
            return model_flat.reshape(image_shape)

        def asymmetric_2d_gaussian_with_poly_bg(params, x, y, degree, image_shape):
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

        def symmetric_2d_gaussian_with_sloped_bg(coords, A, x0, y0, sigma, B0, B1, B2):
            x, y = coords
            gauss = A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            background = B0 + B1 * x + B2 * y
            return gauss + background

        def asymmetric_2d_gaussian_with_sloped_bg(coords, A, x0, y0, sigma_x, sigma_y, theta, B0, B1, B2):
            x, y = coords
            x0 = float(x0)
            y0 = float(y0)
            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = (-np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            gauss = A * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
            background = B0 + B1 * x + B2 * y
            return gauss + background

        def guess_p0_general(img, model_type="symmetric", bg_type="offset", degree=1):
            H, W = img.shape
            Y, X = np.indices((H, W))
            min_val = np.min(img)
            max_val = np.max(img)
            median_val = np.median(img)

            is_dip = (min_val < median_val)
            A = max_val
            y0, x0 = int(H/2), int(W/2)

            if model_type == "symmetric":
                sigma = min(H, W) / 6.0
                gaussian_params = [A, x0, y0, sigma]
                lower_bounds_gauss = [0, 0, 0, -np.inf]
                upper_bounds_gauss = [+np.inf, H, W, +np.inf]

            elif model_type == "asymmetric":
                sigma_x = W / 6.0
                sigma_y = H / 6.0
                theta = 0.0
                gaussian_params = [A, x0, y0, sigma_x, sigma_y, theta]
                lower_bounds_gauss = [0, 0, 0, -np.inf, -np.inf, -np.inf]
                upper_bounds_gauss = [+np.inf, H, W, +np.inf, +np.inf, +np.inf]

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            if bg_type == "none":
                bg_params = []

            elif bg_type == "offset":
                offset = median_val
                bg_params = [offset]
                lower_bounds_bg = [0]
                upper_bounds_bg = [+np.inf]

            elif bg_type == "sloped":
                # flat plane: offset + slope in x and y
                offset = median_val
                B1 = 0.0
                B2 = 0.0
                bg_params = [offset, B1, B2]
                lower_bounds_bg = [0, -np.inf, -np.inf]
                upper_bounds_bg = [+np.inf, +np.inf, +np.inf]

            elif bg_type == "poly":
                # polynomial coefficients + offset
                offset = median_val
                n_terms = (degree + 1) * (degree + 2) // 2  # triangular number
                poly_coeffs = [0.0] * n_terms
                bg_params = [offset] + poly_coeffs
                lower_bounds_bg = [0, [-np.inf] *n_terms]
                upper_bounds_bg = [+np.inf, [+np.inf] *n_terms]

            else:
                raise ValueError(f"Unknown bg_type: {bg_type}")
            print(gaussian_params)
            print(bg_params)
            return gaussian_params + bg_params, (lower_bounds_gauss + lower_bounds_bg, upper_bounds_gauss + upper_bounds_bg)

        available_fit_functions = {"symmetric_2d_gaussian_poly_bg": symmetric_gaussian_2d_with_poly_bg,
                               "asymmetric_2d_gaussian_poly_bg": asymmetric_2d_gaussian_with_poly_bg,
                               "symmetric_2d_gaussian_sloped_bg": symmetric_2d_gaussian_with_sloped_bg,
                               "asymmetric_2d_gaussian_sloped_bg": asymmetric_2d_gaussian_with_sloped_bg,
                               "symmetric_2d_gaussian": symmetric_2d_gaussian,
                               "asymmetric_2d_gaussian": asymmetric_2d_gaussian}

        symmetry = re.search(r"(symmetric|asymmetric)", model_func)
        background = re.search(r"(poly|sloped)", model_func)

        model_type = symmetry.group(1) if symmetry else "unknown"
        bg_type = background.group(1) if background else "offset"


        amplitudes = []
        centers = []
        fit_stack = []
        residue_stack = []
        roi_images = []
        i = 0
        for image in self.stack:
            if np.max(image) > 150:
                H, W = image.shape
                x = np.arange(W)
                y = np.arange(H)
                X, Y = np.meshgrid(x, y)

                coords = (X.ravel(), Y.ravel())
                values = image.ravel()

                p0, bounds = guess_p0_general(image, model_type=model_type, bg_type=bg_type, degree=degree)
                print(bounds)
                try:
                    if bounds:
                        popt, pcov = curve_fit(available_fit_functions[model_func], coords, values, p0=p0, bounds=bounds)
                    else:
                        popt, pcov = curve_fit(available_fit_functions[model_func], coords, values, p0=p0, maxfev=50000)
                except Exception as e:
                    popt = []
                    pcov = []

                if popt is not None and len(popt) > 0:
                    print(i)
                    A, x0, y0 = popt[0], popt[1], popt[2]
                    fitted = available_fit_functions[model_func](coords, *popt).reshape(image.shape)
                    residue = image - fitted

                else:
                    A, x0, y0 = None, None, None
                    fitted = None
                    residue = None

                roi_images.append(image)
                fit_stack.append(fitted)
                amplitudes.append(A)
                centers.append((x0, y0))
                residue_stack.append(residue)
                i = i+1
            else:
                return amplitudes, centers, roi_images, fit_stack, residue_stack

        return amplitudes, centers, roi_images, fit_stack, residue_stack



def select_roi_and_extract_signal(stack):
    fig, ax = plt.subplots()
    ax.imshow(stack[10], cmap='gray')
    ax.set_title("Select ROI (drag a box)")

    roi = {}

    def onselect(eclick, erelease):
        roi['x1'], roi['y1'] = int(eclick.xdata), int(eclick.ydata)
        roi['x2'], roi['y2'] = int(erelease.xdata), int(erelease.ydata)
        plt.close()

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1],
                                  minspanx=5, minspany=5, spancoords='pixels',
                                  interactive=True)
    plt.show()

    if not roi:
        raise RuntimeError("No ROI selected")

    x1, x2 = sorted([roi['x1'], roi['x2']])
    y1, y2 = sorted([roi['y1'], roi['y2']])
    roi_slice = stack[:, y1:y2, x1:x2]
    roi_means = [np.mean(slice) for slice in roi_slice]
    return roi_means, roi_slice

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

class ScrollableFitViewer:
    def __init__(self, original_stack, experiment, fitted_stack=None, residue_stack=None,
                 centers=None, title="2D Gaussian Fit Viewer",
                 cmap="viridis", x_vals=None, y_vals=None):

        self.original_stack = original_stack
        self.fitted_stack = fitted_stack
        self.residue_stack = residue_stack
        self.experiment = experiment
        self.centers = centers if centers is not None else [None] * original_stack.shape[0]
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.index = 0
        self.Z = len(original_stack)
        self.cmap = cmap

        if self.experiment == 'gaussian_fit':
            # Create figure with 4 subplots
            self.fig, axs = plt.subplots(1, 5, figsize=(30, 5))
            self.fig.suptitle(title)
            self.ax1, self.ax2, self.ax3, self.ax4, self.ax5 = axs

            self.im1 = self.ax1.imshow(self.original_stack[self.index], cmap=cmap, origin='upper')
            self.ax1.set_title("Original")
            self.ax1.axis('off')

            self.im2 = self.ax2.imshow(self.fitted_stack[self.index], cmap=cmap, origin='upper')
            self.ax2.set_title("Fitted")
            self.ax2.axis('off')

            self.im3 = self.ax3.imshow(self.residue_stack[self.index], cmap=cmap, origin='upper')
            self.ax3.set_title("Residue")
            self.ax3.axis('off')

            self.cross1 = self.ax1.plot([], [], 'r+', markersize=8)[0]
            self.cross2 = self.ax2.plot([], [], 'r+', markersize=8)[0]

            if self.x_vals is not None and self.y_vals is not None:
                self.line, = self.ax4.plot(self.x_vals, self.y_vals, 'b.-', picker=5)
                self.selected_point, = self.ax4.plot([], [], 'ro', markersize=10)  # highlight selected
                self.ax4.set_title("Click to select slice")
                self.ax4.set_xlabel("X")
                self.ax4.set_ylabel("Y")
                self.ax4.set_ylim(bottom=0)

            if centers is not None:
                center_x, center_y = zip(*centers)
                time = np.arange(len(centers))
                scatter = self.ax5.scatter(center_x, center_y, c=time, cmap='viridis', s=60, edgecolor='k')
                self.ax5.set_xlabel("X Position")
                self.ax5.set_ylabel("Y Position")
                self.ax5.set_aspect('equal')  # optional: keeps x/y scale proportional

                cbar = self.fig.colorbar(scatter, ax=self.ax5)
                cbar.set_label("Time")


            self.update_crosshair()
            self.update_display()

            self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
            self.fig.canvas.mpl_connect("pick_event", self.on_pick)
            plt.tight_layout()
            plt.show()

        elif self.experiment == 'cilia':
            self.fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            self.fig.suptitle(title)
            self.ax1, self.ax4 = axs

            self.im1 = self.ax1.imshow(self.original_stack[self.index], cmap=cmap, origin='upper')
            self.ax1.set_title("Original")
            self.ax1.axis('off')

            if self.x_vals is not None and self.y_vals is not None:
                self.line, = self.ax4.plot(self.x_vals, self.y_vals, 'b.-', picker=5)
                self.selected_point, = self.ax4.plot([], [], 'ro', markersize=10)  # highlight selected
                self.ax4.set_title("Click to select slice")
                self.ax4.set_xlabel("X")
                self.ax4.set_ylabel("Y")
                self.ax4.set_ylim(bottom=0)
                self.ax4.set_ylim(top=np.max(image_stack))

            self.update_display()

            self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
            self.fig.canvas.mpl_connect("pick_event", self.on_pick)
            plt.tight_layout()
            plt.show()

        else:
            raise RuntimeError('No valid experiment type selected.')


    def update_crosshair(self):
        cx, cy = self.centers[self.index] if self.centers[self.index] else (None, None)
        if cx is not None and cy is not None:
            self.cross1.set_data([cx], [cy])
            self.cross2.set_data([cx], [cy])
        else:
            self.cross1.set_data([], [])
            self.cross2.set_data([], [])

    def update_display(self):
        if self.experiment == 'gaussian_fit':
            self.im1.set_data(self.original_stack[self.index])
            self.im2.set_data(self.fitted_stack[self.index])
            self.im3.set_data(self.residue_stack[self.index])

            self.ax1.set_title(f"Original (Slice {self.index})")
            self.ax2.set_title(f"Fitted (Slice {self.index})")
            self.ax3.set_title(f"Residue (Slice {self.index})")

            if self.x_vals is not None and self.y_vals is not None:
                self.selected_point.set_data([self.x_vals[self.index]], [self.y_vals[self.index]])

            self.update_crosshair()
            self.fig.canvas.draw_idle()
        elif self.experiment == 'cilia':
            self.im1.set_data(self.original_stack[self.index])
            self.ax1.set_title(f"Original (Slice {self.index})")

            if self.x_vals is not None and self.y_vals is not None:
                self.selected_point.set_data([self.x_vals[self.index]], [self.y_vals[self.index]])

            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            self.index = (self.index + 1) % self.Z
        elif event.button == 'down':
            self.index = (self.index - 1) % self.Z
        self.update_display()

    def on_pick(self, event):
        # Identify clicked point from the line plot
        ind = event.ind[0]
        self.index = ind
        self.update_display()




path = '/Users/sophia.betzler/Desktop/images'
#### SETUP FOR THE BEADS EXPERIMENT ###########
# image_stack = import_images_from_folder(path)
# roi_means, roi_slice = select_roi_and_extract_signal(image_stack[1:2000])
# plt.plot(roi_means)
# fit = FitFunctionsForStack(roi_slice)
# amplitudes, centers, roi_images, fitted_stack, residue_stack = fit.fit_gaussian_model("symmetric_2d_gaussian", degree=3)
#
# viewer = ScrollableFitViewer(roi_images, 'gaussian_fit', fitted_stack, residue_stack, centers=centers, x_vals=np.arange(len(roi_images)), y_vals=amplitudes)

#### SETUP FOR THE CILIA EXPERIMENT ###########
image_stack = import_images_from_folder(path)
roi_means, roi_slice = select_roi_and_extract_signal(image_stack[1:1400])
viewer = ScrollableFitViewer(roi_slice, experiment='cilia', x_vals=np.arange(len(roi_slice)), y_vals=roi_means)