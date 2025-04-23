#load_ext autoreload
#autoreload 2
#matplotlib inline

from fibsem import utils, acquire, alignment
from fibsem.structures import BeamType, FibsemRectangle
import numpy as np
import matplotlib.pyplot as plt

microscope, settings = utils.setup_session()

# set scan rotation
microscope.set("scan_rotation", np.radians(180), beam_type=BeamType.ION)

# reset beam shifts
microscope.reset_beam_shifts()

# acquire FIB Image
settings.image.hfw = 150e-6
settings.image.beam_type = BeamType.ION
settings.image.reduced_area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)
image1 = acquire.acquire_image(microscope=microscope, settings=settings.image)

# shift beam shift
microscope.beam_shift(dx=5e-6, dy=2.5e-6, beam_type=BeamType.ION)

# align beam shift
alignment.multi_step_alignment_v2(microscope, ref_image=image1, beam_type=BeamType.ION, use_autocontrast=True)

# acquire FIB Image
image2 = acquire.acquire_image(microscope=microscope, settings=settings.image)

# plot
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(image1.data, cmap='gray')
ax[0].set_title('Previous Image')
ax[0].axis('off')
ax[1].imshow(image2.data, cmap='gray')
ax[1].set_title('Post Image')
ax[1].axis('off')

# plot center crosshair
ax[0].plot([image1.data.shape[1] // 2, image1.data.shape[1] // 2], [0, image1.data.shape[0]], color='yellow', lw=1)
ax[0].plot([0, image1.data.shape[1]], [image1.data.shape[0] // 2, image1.data.shape[0] // 2], color='yellow', lw=1)
ax[1].plot([image2.data.shape[1] // 2, image2.data.shape[1] // 2], [0, image2.data.shape[0]], color='yellow', lw=1)
ax[1].plot([0, image2.data.shape[1]], [image2.data.shape[0] // 2, image2.data.shape[0] // 2], color='yellow', lw=1)
plt.show()
