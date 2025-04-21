/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, name: str) -> None:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _on_done(self, done):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, name: str, scintillator_number: int, coordinates=acqstream.UNDEFINED_ROI, colour="#FFA300"):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def estimate_acquisition_time(roa, pre_calibrations=None, acq_dwell_time: Optional[float] = None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def acquire(roa, path, username, scanner, multibeam, descanner, detector, stage, scan_stage, ccd, beamshift, lens,
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, scanner, multibeam, descanner, detector, stage, scan_stage, ccd, beamshift, lens, se_detector,
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def acquire_roa(self, dataflow):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def pre_calibrate(self, pre_calibrations):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def image_received(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def get_pos_first_tile(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def get_abs_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        # Acceleration unknown, guessActuatorMoveDuration uses a default acceleration
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def move_stage_to_next_tile(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def correct_beam_shift(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _create_acquisition_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _calculate_beam_shift_cor_indices(self, n_beam_shifts=10):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def acquireTiledArea(stream, stage, area, live_stream=None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def estimateTiledAcquisitionTime(stream, stage, area, dwell_time=None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, future: model.ProgressiveFuture) -> None:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _cancel_acquisition(self, future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def run(self, stream, stage, area, live_stream):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        def _pass_future_progress(sub_f, start, end):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_rectangular(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_rectangular(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_number_of_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_generate_indices(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_move_to_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_fov(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_area(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_compressed_stack(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_whole_procedure(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_refocus(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_z_on_focus_plane(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_triangulated_focus_point(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_always_focusing_method(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_estimate_time(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def _position_listener(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_registrar_weaver(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_stream_based_bbox(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_tiled_bboxes(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_fov_based_bbox(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_clip_tiling_bbox_to_range(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_zstack_levels(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_real_images_shift(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_real_images_identity(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_dep_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_one_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_no_seam(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:def decompose_image(img, overlap=0.1, numTiles=5, method="horizontalLines", shift=True):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_one_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_real_perfect_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_synthetic_perfect_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_rotated_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_stage_bare_pos(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_brightness_adjust(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_gradient(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_foreground_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def __init__(self, adjust_brightness=False):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def addTile(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def getFullImage(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def get_bounding_boxes(tiles: list):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def get_final_metadata(self, md: dict) -> dict:
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def _adjust_brightness(self, tile, tiles):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_simple.py:def register(tiles, method=REGISTER_GLOBAL_SHIFT):
/home/meteor/development/odemis/src/odemis/acq/stitching/_simple.py:def weave(tiles, method=WEAVER_MEAN, adjust_brightness=False):
Binary file /home/meteor/development/odemis/src/odemis/acq/stitching/__pycache__/_tiledacq.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stitching/__pycache__/_tiledacq.cpython-38.pyc matches
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def __init__(self, streams, stage, area, overlap, settings_obs=None, log_path=None, future=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:           If focus_points is defined, zlevels is adjusted relative to the focus_points.
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:                    # Triangulation needs minimum three points to define a plane
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getFov(self, sd):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _guessSmallestFov(self, ss):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getNumberOfTiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _generateScanningIndices(self, rep):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _moveToTile(self, idx, prev_idx, tile_size):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _sortDAs(self, das, ss):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        def leader_quality(da):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _updateFov(self, das, sfov):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _estimateStreamPixels(self, s):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimateMemory(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimateTime(self, remaining=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:            # move_speed is a default speed but not an actual stage speed due to which
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _save_tiles(self, ix, iy, das, stream_cube_id=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        def save_tile(ix, iy, das, stream_cube_id=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireStreamCompressedZStack(self, i, ix, iy, stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireStreamTile(self, i, ix, iy, stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getTileDAs(self, i, ix, iy):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireTiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_z_on_focus_plane(self, x, y):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_triangulated_focus_point(self, x, y):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _refocus(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_zstack_levels(self, focus_value):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _adjustFocus(self, das, i, ix, iy):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _stitchTiles(self, da_list):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateTiledAcquisitionTime(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateTiledAcquisitionMemory(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def acquireTiledArea(streams, stage, area, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateOverviewTime(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def acquireOverview(streams, stage, areas, focus, detector, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        Each area is defined by 4 floats :left, top, right, bottom positions
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def __init__(self, streams, stage, areas, focus, detector, future=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimate_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _pass_future_progress(self, future, start: float, end: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_stream_based_bbox(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        # fall back to a small fov (default)
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_tiled_bboxes(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_fov_based_bbox(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    :param fov: the fov to acquire, default is DEFAULT_FOV
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_fov(s: Stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def clip_tiling_bbox_to_range(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_zstack_levels(zsteps: int, zstep_size: float, rel: bool = False, focuser=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    :param rel: If this is False (default), then z stack levels are in absolute values. If rel is set to True then
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _insert_tile_to_grid(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _find_closest_tile(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _updateGrid(self, direction):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _estimateROI(self, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _estimateMatch(self, imageA, imageB, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _get_shift(self, prev_tile, tile, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _register_horizontally(self, row, col, xdir):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _register_vertically(self, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _compute_registration(self, tile, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _insert_tile_to_grid(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _get_shift(self, prev_tile, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _compute_registration(self, tile, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _assemble_mosaic(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _check_square_pixel(self, st):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_roi_rep_pxs_links(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_camera_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_histogram(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_hwvas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_weakref(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_default_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        Check the default values for the FluoStream are fitting the current HW
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_sem(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_hwvas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_live_conf(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_progressive_future(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_sync_future_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_ar(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_fuz(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_mn(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_count(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_only(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage_all(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_scan_stage_wrapper(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_scan_stage_wrapper_noccd(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_roi_out_of_stage_limits(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_done(self, _):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_progress_update(self, _, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_live_stream(self):  # TODO  this one has still exposureTime
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streakcam_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_gui_vas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_stream_va_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_live_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_integrated_images_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_se_ebic(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_arpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_arpol_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_stream_va_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_acq_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_acq_integrated_images_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spec_light_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_light(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ebic_live_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ebic_acq(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_live_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spec_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_repetitions_and_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cancel_active(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mnchr_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cl_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cl(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_small_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_int_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_uint16_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_uint16_hist_white(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_ar_data(self, shape, tweak=0, pol=None):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_das(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_allpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_1pol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpolarimetry(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_large_image(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_spectrum_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_das(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_2d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_0d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_1d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_temporal_spectrum_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum_false_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_chronograph_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_chronograph(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_chronograph_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_temporal_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_chronograph(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_tiled_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream_pan(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def getTileMock(self, x, y, zoom):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream_zoom(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def getTileMock(self, x, y, zoom):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_updatable_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_pixel_coordinates(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:def roi_to_phys(repst):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_spot_alignment(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_spot_alignment_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def on_progress_update(self, future, past, left):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_aligned_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:By default, it listens on 127.0.0.1:6000 for a connection. Just launch the EXE with default settings
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def validate_scan_sim(self, das, repetition, dwelltime, pixel_size):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_flim_acq_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def _validate_rep(self, s):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_repetitions_and_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_cancelling(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_rep(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_small_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_tcd_live(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:        # TODO: by default the simulator only sends data for 5s... every 1.5s => 3 data
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_rep_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_entries(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_get_streams_settings(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_set_streams_settings(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_set_streams_settings_missing_keys(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:        self.assertEqual(s.detExposureTime.value, 0.89)  # As defined in the "Few settings" entry
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_get_settings_order(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_set_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_next_rectangle(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_complete(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_shape_1(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def __init__(self, cnvs=None):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def check_point_proximity(self, v_point):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def draw(self, ctx, shift=(0, 0), scale=1.0):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def copy(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def move_to(self, pos):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def get_state(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def restore_state(self, state):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def to_dict(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def from_dict(shape, tab_data):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def set_rotation(self, target_rotation):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def reset(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_overview_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_estimateTiledAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_overview_acquisition_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def on_acquisition_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices_overlap_small_roa(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_acquire_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_coverage_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_progress_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_cancel_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_movement_rotation_correction(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_pos_outside_roa(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_pre_calibrate(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_pos_first_tile(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calibration_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_save_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calculate_beam_shift_cor_indices(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_save_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        def _image_received(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_pre_calibrate(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calibration_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_align_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_cancel_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_current_aligner_position(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def check_move_aligner_to_target(self, target):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_current_position(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_smaract_stage_fallback_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_sem_imaging_area_after_loading_1st_method(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the sem imaging area, and grid1 will be chosen by default.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the fm imaging area, and grid1 will be chosen by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_sem_imaging_area_after_loading_2nd_method(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to grid1, and sem imaging area will be chosen by default.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the fm imaging area, and grid1 will be chosen by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid2_in_sem_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid1_to_grid2_in_sem_imaging_area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid2_to_grid1_in_sem_imaging_area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_sem_to_fm(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid1_to_grid2_in_fm_imaging_Area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # now the grid is grid1 by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid2_to_grid1_in_fm_imaging_Area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_sem_from_fm(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_transformFromSEMToMeteor(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # the grid positions in the metadata of the stage component are defined without rz axes.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_log_linked_stage_positions(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_switching_movements(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def _test_3d_transformations(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_to_posture(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_transformation_calculation(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_scan_rotation(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_component_metadata_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_switching_consistency(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_cancel_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes_but_without_difference(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes_but_without_common_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_rotation_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_rotation_axes_no_difference(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_rotation_axes_missing_axis(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_no_common_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_lin_rot_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_progress(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_progress_lin_rot(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_isNearPosition(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # test user defined tolerance
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_overview(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_live(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_megafield(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def test_feature_acquisitions(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def test_feature_posture_positions(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_encoder(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_decoder(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_save_read_features(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_milling_tasks(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_multi(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_full(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_simple_arpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_background(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_compensation(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_full(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_compensate(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_compensate_out(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_angular_spec_compensation(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_write_read_trigger_delays(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def test_overlay_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def test_acq_fine_align(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_getSingleFrame(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_single_frame_acquisition_VA(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_sync_sem_ccd(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_sync_path_guess(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_valid_folding(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_folding_same_detector_name(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_folding_pairs_of_two(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def _on_progress_update(self, f, s, e):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_only_FM_streams_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_only_SEM_streams_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_FM_and_SEM_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_settings_observer_metadata_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:def path_pos_to_phys_pos(pos, comp, axis):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    Convert a generic position definition into an actual position (as reported in comp.position)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    pos (float, str, or tuple): a generic definition of the position
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    axis_def = comp.axes[axis]
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    if hasattr(axis_def, "choices"):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        for key, value in axis_def.choices.items():
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:def assert_pos_as_in_mode(t: unittest.TestCase, comp, mode):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    Check the position of the given component is as defined for the
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    specified mode (for all the axes defined in the specified mode)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        axis_def = comp.axes[axis]
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:            choices = axis_def.choices
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_queue(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_spec_switch_mirror(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        # set the spec-switch mirror to the default retracted position (FAV_POS_DEACTIVE)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        # exclude the check for pos in mode for the spec-switch, it's not in light-in-align mode by default
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_acq_quality(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def assert_pos_align(self, expected_pos_name: str):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def test_drift_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def on_progress_update(self, future, past, left):
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/test/example_drifted.h5 matches
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_acquire(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimate(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_updateScannerSettings(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:        # Check in the order defined in _updateSEMSettings
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimateCorrectionPeriod(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_identical_inputs(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/test/example_input.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/__pycache__/__init__.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/__pycache__/__init__.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def __init__(self, scanner, detector, region, dwell_time, max_pixels=MAX_PIXELS, follow_drift=True):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:        logging.info("Anchor region defined with scale=%s, res=%s, trans=%s",
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimate(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimateCorrectionPeriod(period, t, repetitions):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def _updateScannerSettings(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:def GuessAnchorRegion(whole_img, sample_region):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:def align_reference_image(
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:# to identify a ROI which must still be defined by the user
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def __init__(self, name, detector, dataflow, emitter, focuser=None, opm=None,
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:          at initialisation. By default, it will contain no data.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # Don't call at init, so don't set metadata if default value
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _init_thread(self, period=0.1):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def emitter(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def detector(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def focuser(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def det_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def emt_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def axis_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def __str__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getVAs(self, comp, va_names):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateVAs(self, comp, prefix, va_names):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _va_sync_setter(self, origva, v):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _va_sync_from_hw(self, lva, v):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateAxis(self, axis_name, actuator):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        axis_name (str): the name of the axis to define
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateAxes(self, axis_map):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateVA(self, va, setter=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _index_in_va_order(self, va_entry):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getEmitterVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getDetectorVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _linkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        Link the axes, which are defined as local VA's,
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:                moves.setdefault(actuator, {})[axis_name] = pos
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onAxisMoveDone(self, f):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _update_linked_position(self, act, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _update_linked_axis(self, va_name, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _unlinkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # This default implementation returns the shortest possible time, taking
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setStatus(self, level, message=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateDRange(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _guessDRangeFromDetector(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getDisplayIRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            # default to small value so that it easily fits in the FoV.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _shouldUpdateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _image_thread(wstream, period=0.1):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onAutoBC(self, enabled):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onOutliers(self, outliers):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _recomputeIntensityRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        data (DataArray): the raw data to use, default to .raw[0] - background
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getPixelCoordinates(self, p_pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # uses a different definition of shear.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getBoundingBox(self, im=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            ValueError: If the stream has no (spatial) data and stream's image is not defined
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            raise ValueError("Cannot compute bounding-box as stream has no data and stream's image is not defined")
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getRawMetadata(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def force_image_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, forcemd=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def getSingleFrame(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        Overwritten by children if they have a dedicated method to get a single frame otherwise the default
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            def on_new_data(future):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _shouldUpdateHistogram(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            logging.debug("Cannot find a scale defined for the %s, using (1, 1) instead" % self.name)
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        Doesn't do anything if no writable resolution/translation VA's are defined on the scanner.
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onDwellTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onResolution(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            logging.debug("Cannot find a scale defined for the %s, using (1, 1) instead" % self.name)
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onMove(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _DoPrepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:#     def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_pxsize(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startSpot(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _stopSpot(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, emtvas=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _stop_light(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # the stream, so use the definition: pxs = sensor pxs * binning / mag
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, emtvas=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _getCount(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, em_filter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # be selected. The difficulty comes to pick the default value. We try
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # default value. In that case, we pick the emission value that matches
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def onExcitation(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def onEmission(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _get_current_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_emission(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _protect_detector(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_streak_mode(self, streak: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_mcp_gain(self, gain: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # We use resolution as a shortcut to define the pitch between pixels
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onZoom(self, z):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onResolution(self, r):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setResolution(self, res):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, scanner, em_filter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def scanner(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def setting_stream(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _getScannerVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _recomputeIntensityRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            # TODO: allows to define the power via a VA on the stream
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _shouldUpdateHistogram(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _clean_raw(self, raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                    logging.warning(u"Metadata for 3D data invalid. Using default pixel size 10µm")
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                    logging.warning(u"Metadata for 3D data invalid. Using default centre position 0")
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _on_zIndex(self, val):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _on_max_projection(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        default_tint = (0, 255, 0)  # green is most typical
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                default_tint = conversion.wavelength2rgb(numpy.mean(em_range))
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        tint = raw.metadata.get(model.MD_USER_TINT, default_tint)
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            self.tint.value = default_tint
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, data, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            def dis_bbtl(v):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setBackground(self, bg_data):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, image, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        default_dims = "CTZYX"
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            default_dims = "CAZYX"
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        dims = image.metadata.get(model.MD_DIMS, default_dims[-image.ndim::])
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateDRange(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setAngle(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setWavelength(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onTimeSelect(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onWavelengthSelect(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setLine(self, line):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _get_bandwidth_in_pixel(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateCalibratedData(self, data=None, bckg=None, coef=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setBackground(self, bckg):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setEffComp(self, coef):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _force_selected_spectrum_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onCalib(self, unused):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onSelectionWidth(self, width):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def onSpectrumBandwidth(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _clean_raw(self, raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def update(self, raw):
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_sync.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_static.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_sync.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/__init__.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_helper.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_live.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_base.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_projection.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/__init__.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_base.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_helper.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_live.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_projection.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_static.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:          The first stream with .repetition will be used to define
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:          The first stream with .useScanStage will be used to define a scanning
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    #     def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def streams(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def leeches(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateProgress(self, future, dur, current, tot, bonus=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _restoreHardwareSettings(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getPixelSize(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getSpotPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getLeftTopPositionPhys(self) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getScanStagePositions(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onHwSyncData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _preprocessData(self, n, data, i):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _get_center_pxs(self, rep: Tuple[int, int],
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assemble2DData(self, rep, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleTiles(self, rep, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleAnchorData(self, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _startLeeches(self, img_time, tot_num, shape):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _stopLeeches(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:         :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveDataTiles(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData2D(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _supports_hw_sync(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettingsHwSync(self) -> Tuple[float, int]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionHwSyncEbeam(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:            # Configure the CCD to the defined exposure time, and hardware sync + get frame duration
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionEbeam(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _waitForImage(self, img_time):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, n, px_idx, img_time, sem_time,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettingsScanStage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionScanStage(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:                        # MD_POS default to the center of the sample stage, but it needs to be the position
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _get_next_rectangle(self, rep: Tuple[int, int], acq_num: int, px_time: float,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n, raw_data, px_idx, px_pos, rep, pol_idx=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:         :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, x, y, img_time):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getNumDriftCors(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n, raw_data, px_idx, px_pos, rep, pol_idx=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:            # for long exposure times. Should be defined on the streak-ccd.metadata[MD_CALIB]["intensity_limit"]
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, n, px_idx, img_time, sem_time,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        # overrides the default _acquireImage to check the light intensity after every image from the
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _check_light_intensity(self, raw_data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        # MD_POS default to position at the center of the FoV, but it needs to be
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, stream, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        It acquires by scanning the defined ROI multiple times. Each scanned
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def streams(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _prepareHardware(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onAcqStop(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _add_frame(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateProgress(self, future, dur, current, tot, bonus=2):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, scanner=None, sstage=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:            # If not integration time, default is 1 image.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # We overwrite the VA provided by LiveStream to define a setter.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setIntegrationTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkIntTime2HwExpTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkIntTime2HwExpTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onIntegrationTime(self, intTime):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _intTime2NumImages(self, intTime):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def scanner(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _getScannerVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onMagnification(self, mag):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _adaptROI(self, roi, rep, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _fitROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _computePixelSize(self, roi, rep):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateROIAndPixelSize(self, roi, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # If ROI is undefined => link rep and pxs as if the ROI was full
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setPixelSize(self, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setRepetition(self, repetition):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # If ROI is undefined => link repetition and pxs as if ROI is full
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _getPixelSizeRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, light=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPower(self, power: float):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setup_light(self, ch_power: Optional[float] = None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _stop_light(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, streak_unit, streak_delay,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.pixelSize.value *= 30  # increase default value to decrease default repetition rate
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _suspend(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _protect_detector(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_streak_mode(self, streak: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_mcp_gain(self, gain: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_center_wavelength(self, wl: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, analyzer=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPolarization(self, pol):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPolarizationMove(self, f):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # (due to the long exposure time). So make the default pixel size bigger.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, spectrometer, spectrograph, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.pixelSize.value *= 30  # increase default value to decrease default repetition rate
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onBinning(self, _=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onMagnification(self, mag):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # For the live view, we need a way to define the scale and resolution,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # current SEM pixelSize/mag/FoV) to define the scale. The ROI is always
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPixelSize(self, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onDwellTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onResolution(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, emt_dataflow, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateAcquisitionTime(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewEmitterData(self, dataflow: model.DataFlow, data: model.DataArray) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateMD(self, md: Dict[str, Any]) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow: model.DataFlow, data: model.DataArray) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _restart_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _set_dwell_time(self, dt: float) -> float:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _set_hw_dwell_time(self, dt: float) -> float:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onResolution(self, value: Tuple[int, int]):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, ccd, emitter, emd, opm=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.repetition = model.ResolutionVA((4, 4),  # good default
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _result_wrapper(self, f) -> Callable:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        def result_as_da(timeout=None) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, emitter, scanner, time_correlator,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        A helper stream used to define FLIM acquisition settings and run a live setting stream
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _image_thread(wprojection):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _shouldUpdateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:        # Don't call at init, so don't set metadata if default value
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _shouldUpdateImageEntirely(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def force_image_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPoint(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getBackground(self, pol_mode):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _processBackground(self, data, pol_mode, clip_data=True):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _resizeImage(self, data, size):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2Polar(self, ebeam_pos, pol_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            polar_data = polar_cache.setdefault(ebeam_pos, {})[pol_pos]
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # define the size of the image for polar representation in GUI
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPolarization(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            output_size = (90, 360)  # Note: increase if data is high def
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsVis(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getRawData(self, ebeam_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectAsRaw(self, ebeam_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                output_size = (400, 600)  # defines the resolution of the displayed image
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                    # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2RGBPolar(self, ebeam_pos, pol_pos, cache_raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # define the size of the image for polar representation in GUI
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPolarimetry(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsVis(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __new__(cls, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _update_rect(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onMpp(self, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onRect(self, rect):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _set_mpp(self, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getPixelCoordinates(self, p_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onZIndex(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_max_projection(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getBoundingBox(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _zFromMpp(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _rectWorldToPixel(self, rect):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getTile(
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectTile(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getTilesFromSelectedArea(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:        Get the tiles inside the region defined by .rect and .mpp
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_tint(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_spectrumBandwidth(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_line(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_time(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_angle(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selection_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selection_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_time(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_angle(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_wl(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_wl(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __init__(self, operator=None, streams=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:            By default operator is an average function.
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __str__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __len__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __getitem__(self, index):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __contains__(self, sp):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def add_stream(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def remove_stream(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def on_stream_update_changed(self, _=None):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def getProjections(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:#     def getImage(self, rect, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:#         #  that will define where to save the result
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def getImages(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def get_projections_by_type(self, stream_types):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def acquire(streams, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def acquireZStack(streams, zlevels, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, future, streams, zlevels, settings_obs):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def cancel(self, _):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def estimate_total_duration(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def estimateZStackAcquisitionTime(streams, zlevels):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def estimateTime(streams):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def foldStreams(streams, reuse=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def computeThumbnail(streamTree, acqTask):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def _weight_stream(stream):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def sortStreams(streams):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, streams, future, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def _adjust_metadata(self, raw_data):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def _on_progress_update(self, f, start, end):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, microscope: model.HwComponent, components: Set[model.HwComponent]):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    # For the .position, the axes definition may contain also the "user-friendly" name
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    def update_settings(value: Dict[str, float], comp_name=comp.name, va_name=va_name, comp=comp):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                            axes_def: Dict[str, model.Axis] = comp.axes
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                if (axis_name in axes_def and hasattr(axes_def[axis_name], "choices")
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    and isinstance(axes_def[axis_name].choices, dict)
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    and p in axes_def[axis_name].choices
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    value[axis_name] = f"{p} ({axes_def[axis_name].choices[p]})"
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    def update_settings(value, comp_name=comp.name, va_name=va_name):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def get_all_settings(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/fastem.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/leech.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/calibration.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/millmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/move.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/calibration.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/leech.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/acqmng.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/acqmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/move.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:def align(scanner, multibeam, descanner, detector, stage, ccd, beamshift, det_rotator,
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:def estimate_calibration_time(calibrations):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def __init__(self, future, scanner, multibeam, descanner, detector, stage, ccd, beamshift, det_rotator,
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:        :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:                # def _pass_future_progress(sub_f, start, end):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def run_calibration(self, calibration, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def estimate_calibration_time(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_image_pair(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_no_match(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/blank_image.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_cancel_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_estimate_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_noise(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_missing_point(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_cosmic_ray(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_noise_missing_point_cosmic_ray(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_single_element(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_distorted_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output_missing_point_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled__distorted_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output_missing_point_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_opt(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_focus_at_boundaries(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_different_starting_positions(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical_multiple_runs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical_start_at_good_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def test_convert_das_to_numpy_stack(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def test_run_tdct_correlation(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_cosmic_ray.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_optical_autofocus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_image_translation_prealign(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_image_dark_gain(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def on_progress_update(self, future, start, end):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/navcam-calib2.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_off_on_spccd(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_off_on_ccd(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_1pix_off_on(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_timeout(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_light_already_on(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_no_detection_of_change(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/transform_test.py:    def test_calculate_transform(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/brightlight-off-slit-ccd.ome.tiff matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/single_part.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/sem_hole.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_spot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_center_spot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def __init__(self, fake_img, aligner):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def _simulate_image(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_close_to_image_edge(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_affine(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_similarity(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_with_shear(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_with_rotation(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_wrong_method(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_on_image(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def _prepare_hardware(self, ebeam_kwargs=None, ebeam_mag=2000, ccd_img=None):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay_failure(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_bad_mag(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_ratio_4_3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_ratio_3_4(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/centers_grid.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/real_optical.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_missing_point.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def _move_to_vacuum(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_hole_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_sh_hole_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_lens_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_no_hole(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_hole_detection(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_update_offset_rot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_align_offset(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_rotation_calculation(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_hfw_shift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_res_shift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:#     def test_scan_pattern(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/one_spot.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_10x10.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_measure_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_opt(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_sem(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_sem_hint(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_det_external(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_spectrograph(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_ded_spectrograph(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_spect(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_slit(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_determine_z_position(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_key_error(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    # def test_module_not_found_error(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_pos_shifted(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_bad_angle(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:        # rng = numpy.random.default_rng(0)
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_identical_inputs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_known_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_random_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_different_precisions(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_identical_inputs_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_known_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_random_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_different_precisions_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_identical_inputs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_random_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_identical_inputs_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_random_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def do_autofocus_in_roi(
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def estimate_autofocus_in_roi_time(n_focus_points, detector, focus, focus_rng, average_focus_time=None):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def _cancel_autofocus_bbox(future):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def autofocus_in_roi(
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/fastem.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/autofocus.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/shift.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/find_overlay.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/shift.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/autofocus.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/find_overlay.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/z_localization.cpython-38.pyc matches
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:    # Sift is not installed by default, check first if it's available
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:# Missing defines from OpenCV
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:def FindTransform(ima, imb, fd_type=None):
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:            search_params = dict(checks=32)  # default value
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:def preprocess(im, invert, flip, crop, gaussian_sigma, eqhis):
/home/meteor/development/odemis/src/odemis/acq/align/transform.py:def CalculateTransform(optical_coordinates, electron_coordinates, skew=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def FindOverlay(repetitions, dwell_time, max_allowed_diff, escan, ccd, detector, skew=False, bgsub=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _DoFindOverlay(future, repetitions, dwell_time, max_allowed_diff, escan,
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:                def dist_to_p1(p):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _CancelFindOverlay(future):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _computeGridRatio(coord, shape):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def estimateOverlayTime(dwell_time, repetitions):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _transformMetadata(optical_image, transformation_values, escan, ccd, skew=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _MakeReport(msg, data, optical_image=None, subimages=None):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _set_blanker(escan, active):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def __init__(self, repetitions, dwell_time, escan, ccd, detector, bgsub=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _save_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _restore_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _discard_data(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _onCCDImage(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _onSpotImage(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _doSpotAcquisition(self, electron_coordinates, scale):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _doWholeAcquisition(self, electron_coordinates, scale):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def DoAcquisition(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def CancelAcquisition(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def get_sem_fov(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def get_ccd_fov(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def sem_roi_to_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:        Converts a ROI defined in the SEM referential a ratio of FoV to a ROI
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:        logging.info("ROI defined at %s m", phys_rect)
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def convert_roi_ratio_to_phys(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def convert_roi_phys_to_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def configure_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def huang(z, calibration_data):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def thunderstorm(z, calibration_data):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:    This formula is based on the default algorithm used in the Thunderstorm ImageJ plugin as described in the reference
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def solve_psf(z, obs_x, obs_y, calibration_data, model_function=huang):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def determine_z_position(image, calibration_data, fit_tol=0.1):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:                        4 = Outputted Z position is outside the defined maximum range from the calibration, output is inaccurate
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def measure_z(stigmator, angle: float, pos: Tuple[float, float], stream, logpath: Optional[str]=None) -> ProgressiveFuture:
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def estimate_measure_z(stigmator, angle: float, pos: Tuple[float, float], stream) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def _do_measure_z(f: ProgressiveFuture, stigmator, angle: float, pos: Tuple[float, float], stream,
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def _cancel_localization(future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def getNextImage(det, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def receive_one_image(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AcquireNoBackground(det, dfbkg=None, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoBinaryFocus(future, detector, emt, focus, dfbkg, good_focus, rng_focus):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:                # so let's "center" on it. That's better than the default behaviour
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoExhaustiveFocus(future, detector, emt, focus, dfbkg, good_focus, rng_focus):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelAutoFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def estimateAcquisitionTime(detector, scanner=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def estimateAutoFocusTime(detector, emt, focus: model.Actuator, dfbkg=None, good_focus=None, rng_focus=None, method=MTD_BINARY) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def Sparc2AutoFocus(align_mode, opm, streams=None, start_autofocus=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    This automatically defines the spectrograph to use and the detectors.
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _cancelSparc2ManualFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def Sparc2ManualFocus(opm, align_mode, toggled=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoSparc2ManualFocus(opm, spgr: model.Actuator, bl: model.Emitter, align_mode: str, toggled: bool = True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def GetSpectrometerFocusingDetectors(focuser: model.Actuator) -> List[model.Detector]:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _findSparc2Focuser(align_mode: str) -> model.Actuator:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _getSpectrometerFocusingComponents(focuser: model.Actuator) -> Tuple[
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _findSameAffects(roles: List[str], affected: List[str]) -> model.Component:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoSparc2AutoFocus(future, streams, align_mode, opm, dets, spgr, selector, bl, focuser, start_autofocus=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def updateProgress(subf, start, end):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelSparc2AutoFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AutoFocus(detector, emt, focus, dfbkg=None, good_focus=None, rng_focus=None, method=MTD_BINARY):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AutoFocusSpectrometer(spectrograph, focuser, detectors, selector=None, streams=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _totalAutoFocusTime(spectrograph, focuser, detectors, selector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _updateAFSProgress(future, af_dur, grating_moves, detector_moves):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def CLSpotsAutoFocus(detector, focus, good_focus=None, rng_focus=None, method=MTD_EXHAUSTIVE):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _mapDetectorToSelector(selector, detectors):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _playStream(detector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoAutoFocusSpectrometer(future, spectrograph, focuser, detectors, selector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def is_current_det(d):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelAutoFocusSpectrometer(future):
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def _convert_das_to_numpy_stack(das: List[model.DataArray]) -> numpy.ndarray:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def get_optimized_z_gauss(das: List[model.DataArray], x: int, y: int, z: int, show: bool = False) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def run_tdct_correlation(fib_coords: numpy.ndarray,
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def get_reprojected_poi_coordinate(correlation_results: dict) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def parse_3dct_yaml_file(path: str) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def list_hw_settings(escan, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def restore_hw_settings(escan, ccd, hw_settings):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def DelphiCalibration(main_data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoDelphiCalibration(future, main_data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _StoreConfig(path, shid, calib_values):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelDelphiCalibration(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateDelphiCalibration():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def AlignAndOffset(ccd, detector, escan, sem_stage, opt_stage, focus, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoAlignAndOffset(future, ccd, detector, escan, sem_stage, opt_stage, focus, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelAlignAndOffset(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateOffsetTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def RotationAndScaling(ccd, detector, escan, sem_stage, opt_stage, focus, offset,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoRotationAndScaling(future, ccd, detector, escan, sem_stage, opt_stage, focus,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:    def find_spot(n):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelRotationAndScaling(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateRotationAndScalingTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def HoleDetection(detector, escan, sem_stage, ebeam_focus, manual=False,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoHoleDetection(future, detector, escan, sem_stage, ebeam_focus, manual=False,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:        def find_sh_hole(holep):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:        def find_focus():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelHoleDetection(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateHoleDetectionTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def FindCircleCenter(image, radius, max_diff, darkest=False):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def FindRingCenter(image):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def UpdateOffsetAndRotation(new_first_hole, new_second_hole, expected_first_hole,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def LensAlignment(navcam, sem_stage, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoLensAlignment(future, navcam, sem_stage, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateLensAlignmentTime():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def HFWShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelFuture(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoHFWShiftFactor(future, detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateHFWShiftFactorTime(et):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def ResolutionShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoResolutionShiftFactor(future, detector, escan, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateResolutionShiftFactorTime(et):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def ScaleShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoScaleShiftFactor(future, detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def turnOnLight(bl, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def _cancelTurnOnLight(f):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def _doTurnOnLight(f, bl, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def getch():
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def print_col(colour, s, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def input_col(colour, s):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def __init__(self, sem_stage, opt_focus, ebeam_focus, opt_stepsize, ebeam_stepsize):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def _move_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def focusByArrow(self, rollback_pos=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def man_calib(logpath, keep_loaded=False):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                    # the range of the tmcm axes to be defined per sample
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                    def ask_user_to_focus(n):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def main(args):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                        default=0, help="set verbosity level (0-2, default = 0)")
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    parser.add_argument("--keep-loaded", dest="keep_loaded", action="store_true", default=False,
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def FindEbeamCenter(ccd, detector, escan):
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def _ConvertCoordinates(coord, img):
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def DivideInNeighborhoods(data, number_of_spots, scale, sensitivity_limit=100):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def ReconstructCoordinates(subimage_coordinates, spot_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def FilterOutliers(image, subimages, subimage_coordinates, expected_spots):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def MatchCoordinates(input_coordinates, electron_coordinates, guess_scale, max_allowed_diff):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _KNNsearch(x_coordinates, y_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _TransformCoordinates(x_coordinates, translation, rotation, scale):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _MatchAndCalculate(transformed_coordinates, optical_coordinates, electron_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:    # Correct for shift if 'too many' points are wrong, with 'too many' defined by:
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _FindOuterOutliers(x_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _FindInnerOutliers(x_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:def _upsampled_dft(data, upsampled_region_size,
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:def MeasureShift(previous_img, current_img, precision=1):
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:    of the maximum of the cross-correlation define the shift vector between the two images.
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def MeasureSNR(image):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def AlignSpot(ccd, stage, escan, focus, type=OBJECTIVE_MOVE, dfbkg=None, rng_f=None, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _DoAlignSpot(future, ccd, stage, escan, focus, type, dfbkg, rng_f, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _CancelAlignSpot(future):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def estimateAlignmentTime(et, dist=None, n_autofocus=2):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _set_blanker(escan, active):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def FindSpot(image, sensitivity_limit=100):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def FindGridSpots(image, repetition, spot_size=18, method=GRID_AFFINE):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:    # Estimate the two most common (orthogonal) directions in the grid of spots, defined in the image coordinate system.
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def CropFoV(ccd, dfbkg=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def CenterSpot(ccd, stage, escan, mx_steps, type=OBJECTIVE_MOVE, dfbkg=None, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _DoCenterSpot(future, ccd, stage, escan, mx_steps, type, dfbkg, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _CancelCenterSpot(future):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def estimateCenterTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def __init__(self, ccd, detector, escan, opt_stage, focus, pattern):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _save_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _restore_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _discard_data(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def DoPattern(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def CancelPattern(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:def wrapSimpleStreamIntoFuture(stream):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def cancel(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def _run(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def _image_listener(self, image):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def __init__(self, future: Future, tasks: List[MillingTaskSettings], fib_stream: FIBStream, filename: str = None):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def cancel(self, future: Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def estimate_milling_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run_milling(self, settings: MillingTaskSettings):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:            self.fibsem.set_default_patterning_beam_type(milling_channel)
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def run_milling_tasks(tasks: List[MillingTaskSettings], fib_stream: FIBStream, filename: str = None) -> Future:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def get_associated_tasks(wt: MillingWorkflowTask,
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def __init__(self,
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def cancel(self, future: Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def check_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _move_to_milling_position(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _run_milling_tasks(self, feature: CryoFeature, workflow_task: MillingWorkflowTask) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _align_reference_image(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _acquire_reference_images(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def get_filename(self, feature: CryoFeature, basename: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def run_automated_milling(features: List[CryoFeature],
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def create_openfibsem_microscope() -> 'OdemisMicroscope':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    # loads the default config
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_pattern_to_openfibsem(p: MillingPatternParameters) -> 'BasePattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_rectangle_pattern(p: RectanglePatternParameters) -> 'RectanglePattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_trench_pattern(p: TrenchPatternParameters) -> 'TrenchPattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_microexpansion_pattern(p: MicroexpansionPatternParameters) -> 'MicroExpansionPattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_milling_settings(s: MillingSettings) -> 'FibsemMillingSettings':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_task_to_milling_stage(task: MillingTaskSettings) -> 'FibsemMillingStage':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_milling_tasks_to_milling_stages(milling_tasks: List[MillingTaskSettings]) -> List['FibsemMillingStage']:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def __init__(self, future: futures.Future,
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def cancel(self, future: futures.Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def estimate_milling_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def run_milling(self, stage: 'FibsemMillingStage') -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def run_milling_tasks_openfibsem(tasks: List[MillingTaskSettings],
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_milling_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_milling_task_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_save_load_task_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_automated_milling(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_automated_milling_doesnt_start(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_cancel_workflow(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_trench_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_rectangle_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_microexpansion_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def draw_milling_tasks(image: model.DataArray, milling_tasks: Dict[str, MillingTaskSettings]) -> plt.Figure:
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/tasks.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/tasks.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/millmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/patterns.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/patterns.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:This module contains structures to define milling tasks and parameters.
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __init__(self, current: float, voltage: float, field_of_view: float, mode: str = "Serial", channel: str = "ion", align: bool = True):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def from_dict(data: dict) -> "MillingSettings":
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __init__(self, milling: dict, patterns: List[MillingPatternParameters], name: str):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def from_dict(data: dict) -> "MillingTaskSettings":
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:def save_milling_tasks(path: str, milling_tasks: Dict[str, MillingTaskSettings]) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:def load_milling_tasks(path: str) -> Dict[str, MillingTaskSettings]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:This module contains structures to define milling patterns.
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, name: str):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List['MillingPatternParameters']:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, rotation: float = 0.0, center = (0, 0), scan_direction: str = "TopToBottom", name: str = "Rectangle"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'RectanglePatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, spacing: float, center = (0, 0), name: str = "Trench"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'TrenchPatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, spacing: float, center = (0, 0), name: str = "Trench"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'MicroexpansionPatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/path.py:                 # of picking a grating (non mirror) by default, using a "local axis"
/home/meteor/development/odemis/src/odemis/acq/path.py:            'ek-align': (r"ccd.*",  # Same as "ek", but with grating and filter explicitly set by default
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def submit(self, fn, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/path.py:def affectsGraph(microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __new__(cls, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:            # Consider that by default we are in "fast" acquisition, with the fan
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _getComponent(self, role):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def setAcqQuality(self, quality):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def setPath(self, mode, detector=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _doSetPath(self, path, detector):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def selectorsToPath(self, target):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _compute_role_to_mode(self, modes: Dict[str, Tuple]) -> Tuple[Tuple[str, str]]:
/home/meteor/development/odemis/src/odemis/acq/path.py:    def guessMode(self, guess_stream):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def getStreamDetector(self, path_stream):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def findNonMirror(self, choices):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def mdToValue(self, comp, md_name):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def affects(self, affecting, affected):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def findPath(self, node1, node2, path=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _setCCDFan(self, enable):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _waitTemperatureReached(self, comp, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _doSetPath(self, path, detector):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __new__(cls, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:            # Check the version in MD_CALIB, defaults to tfs_1
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def cryoSwitchSamplePosition(self, target: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target_pos: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getMovementProgress(self, current_pos: Dict[str, float], start_pos: Dict[str, float],
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getDistance(self, start: dict, end: dict) -> float:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_stage_metadata(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _cancelCryoMoveSample(self, future):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _run_reference(self, future, component):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _run_sub_move(self, future, component, sub_move):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _update_posture(self, position: Dict[str, float]):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def at_milling_posture(self, pos: Dict[str, float], stage_md: Dict[str, float]) -> bool:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def get_posture_orientation(self, posture: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _initialise_transformation(
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_rot_matrix(self, invert: bool = False) -> RigidTransform:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_sample_from_stage(self, val: List[float], absolute=True) -> List[float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_sample_to_stage(self, val: List[float], absolute=True) -> List[float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _update_conversion(self):
/home/meteor/development/odemis/src/odemis/acq/move.py:        # NOTE: transformations are defined as stage -> sample stage
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_scan_rotation_matrix(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_stage_pos(self, sample_val: Dict[str, float], absolute: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_sample_pos(self, stage_val: Dict[str, float], absolute: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def constrain_stage_pos_axes(self, stage_val: Dict[str, float], fixed_sample_axes: Dict[str, float],
/home/meteor/development/odemis/src/odemis/acq/move.py:        Get the stage (bare) coordinates by constraining the sample plane to a defined plane given by fixed_sample_pos.
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_to_sample_stage_from_stage(self, pos_dep, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_from_sample_stage_to_stage(self, pos, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_pos_vector(self, pos_val, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_movement(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_position(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_sample_stage_from_stage_position(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_movement2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_position2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_sample_stage_from_stage_position2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_posture(self, pos: Dict[str, float], posture: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_sem_to_milling(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_milling_to_sem(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fm_to_milling(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_milling_to_fm(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:        # this is not the case by default, and needs to be added in the metadata
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:        # defined by the Z axis of the sample plane. Default value is computed by GRID 1 centre of stage (bare) in FM
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_fm_sample_z(self, stage_pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float], fix_fm_plane: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def create_sample_stage(self):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fib_to_fm(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fm_to_fib(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:            # axis with choices defining the position to "parked" and "engaged".
/home/meteor/development/odemis/src/odemis/acq/move.py:                # If not in imaging mode yet, move the stage to the default imaging position
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getCurrentStagePostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def get3beamsSafePos(self, active_pos: Dict[str, float], safety_margin: float) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _cryoSwitchAlignPosition(self, target: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchAlignPosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getCurrentAlignerPositionLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:        # TODO: should have a POS_ACTIVE_RANGE to define the whole region
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, name: str, role: str, stage_bare: model.Actuator , posture_manager: MicroscopePostureManager, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _updatePosition(self, pos_dep):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _updateSpeed(self, dep_speed):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveRel(self, shift: Dict[str, float], **kwargs) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveAbs(self, pos: Dict[str, float], **kwargs) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveRelChamberReferential(self, shift: Dict[str, float]) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def stop(self, axes=None):
/home/meteor/development/odemis/src/odemis/acq/move.py:def calculate_stage_tilt_from_milling_angle(milling_angle: float, pre_tilt: float, column_tilt: int = math.radians(52)) -> float:
/home/meteor/development/odemis/src/odemis/acq/move.py:    :param column_tilt: the column tilt in radians (default TFS = 52deg, Tescan = 55deg)
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_scanner(scanner, mode):
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_detector(detector, roc2, roc3):
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_multibeam(multibeam):
/home/meteor/development/odemis/src/odemis/acq/leech.py:def get_next_rectangle(shape, current, n):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_start(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self, scanner, detector):
/home/meteor/development/odemis/src/odemis/acq/leech.py:        # in seconds, default to "fairly frequent" to work hopefully in most cases
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def tot_drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def max_drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _setROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_start(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:            raise ValueError("AnchorDriftCorrector.roi is not defined")
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self, detector, selector=None):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _acquire(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(self, name: str,
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def set_posture_position(self, posture: str, position: Dict[str, float]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def get_posture_position(self, posture: str) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def save_milling_task_data(self,
/home/meteor/development/odemis/src/odemis/acq/feature.py:def get_feature_position_at_posture(pm: MicroscopePostureManager,
/home/meteor/development/odemis/src/odemis/acq/feature.py:def get_features_dict(features: List[CryoFeature]) -> Dict[str, str]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def object_hook(self, obj):
/home/meteor/development/odemis/src/odemis/acq/feature.py:def save_features(project_dir: str, features: List[CryoFeature]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def read_features(project_dir: str) -> List[CryoFeature]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def load_project_data(path: str) -> dict:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def add_feature_info_to_filename(feature: CryoFeature, filename: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def _create_fibsem_filename(filename: str, acq_type: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def cancel(self, future: futures.Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _acquire_feature(self, site: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _run_autofocus(self, site: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _move_to_site(self, site: CryoFeature):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _move_focus(self, site: CryoFeature, fm_focus_position: Dict[str, float]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _generate_zlevels(self, zmin: float, zmax: float, zstep: float) -> Dict[Stream, List[float]]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_acquisition_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_movement_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_total_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def run(self) -> Tuple[List[model.DataArray], Exception]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _export_data(self, feature: CryoFeature, data: List[model.DataArray]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def acquire_at_features(
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:def get_settings_order(stream):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def __init__(self, file_path, max_entries):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _read(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _write(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def entries(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _get_config_index(self, config_data: list, name: str) -> int:
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def update_data(self, new_entries: List[Dict[str, Any]]):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def update_entries(self, streams: list):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def apply_settings(self, stream: Stream, name: str):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_ar_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:            ar_data.setdefault(polmode, []).append(da)
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_spectrum_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_temporalspectrum_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_spectrum_efficiency(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def apply_spectrum_corrections(data, bckg=None, coef=None):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def project_angular_spectrum_to_grid_5d(data: model.DataArray) -> model.DataArray:
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def write_trigger_delay_csv(filename, trig_delays):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def read_trigger_delay_csv(filename, time_choices, trigger_delay_range):
meteor@meteor5031:/$ grep -r 'def' '/home/meteor/development/odemis/src/odemis/acq'
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, name: str) -> None:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _on_done(self, done):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, name: str, scintillator_number: int, coordinates=acqstream.UNDEFINED_ROI, colour="#FFA300"):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def estimate_acquisition_time(roa, pre_calibrations=None, acq_dwell_time: Optional[float] = None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def acquire(roa, path, username, scanner, multibeam, descanner, detector, stage, scan_stage, ccd, beamshift, lens,
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, scanner, multibeam, descanner, detector, stage, scan_stage, ccd, beamshift, lens, se_detector,
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def acquire_roa(self, dataflow):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def pre_calibrate(self, pre_calibrations):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def image_received(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def get_pos_first_tile(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def get_abs_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        # Acceleration unknown, guessActuatorMoveDuration uses a default acceleration
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def move_stage_to_next_tile(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def correct_beam_shift(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _create_acquisition_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _calculate_beam_shift_cor_indices(self, n_beam_shifts=10):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def acquireTiledArea(stream, stage, area, live_stream=None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:def estimateTiledAcquisitionTime(stream, stage, area, dwell_time=None):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def __init__(self, future: model.ProgressiveFuture) -> None:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def _cancel_acquisition(self, future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/fastem.py:    def run(self, stream, stage, area, live_stream):
/home/meteor/development/odemis/src/odemis/acq/fastem.py:        def _pass_future_progress(sub_f, start, end):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_rectangular(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_white_image(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_real_manual(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_shift_rectangular(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/registrar_test.py:    def test_dependent_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_number_of_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_generate_indices(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_move_to_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_fov(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_area(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_compressed_stack(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_whole_procedure(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_refocus(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_z_on_focus_plane(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_triangulated_focus_point(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_always_focusing_method(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_estimate_time(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def _position_listener(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_registrar_weaver(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_stream_based_bbox(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_tiled_bboxes(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_fov_based_bbox(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_clip_tiling_bbox_to_range(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/tiledacq_test.py:    def test_get_zstack_levels(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_real_images_shift(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_real_images_identity(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_dep_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_one_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:    def test_no_seam(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/stitching_test.py:def decompose_image(img, overlap=0.1, numTiles=5, method="horizontalLines", shift=True):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_one_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_real_perfect_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_synthetic_perfect_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_rotated_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_stage_bare_pos(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_brightness_adjust(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_gradient(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/stitching/test/weaver_test.py:    def test_foreground_tile(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def __init__(self, adjust_brightness=False):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def addTile(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def getFullImage(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def get_bounding_boxes(tiles: list):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def get_final_metadata(self, md: dict) -> dict:
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def _adjust_brightness(self, tile, tiles):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_weaver.py:    def weave_tiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_simple.py:def register(tiles, method=REGISTER_GLOBAL_SHIFT):
/home/meteor/development/odemis/src/odemis/acq/stitching/_simple.py:def weave(tiles, method=WEAVER_MEAN, adjust_brightness=False):
Binary file /home/meteor/development/odemis/src/odemis/acq/stitching/__pycache__/_tiledacq.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stitching/__pycache__/_tiledacq.cpython-38.pyc matches
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def __init__(self, streams, stage, area, overlap, settings_obs=None, log_path=None, future=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:           If focus_points is defined, zlevels is adjusted relative to the focus_points.
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:                    # Triangulation needs minimum three points to define a plane
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getFov(self, sd):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _guessSmallestFov(self, ss):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getNumberOfTiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _generateScanningIndices(self, rep):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _moveToTile(self, idx, prev_idx, tile_size):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _sortDAs(self, das, ss):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        def leader_quality(da):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _updateFov(self, das, sfov):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _estimateStreamPixels(self, s):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimateMemory(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimateTime(self, remaining=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:            # move_speed is a default speed but not an actual stage speed due to which
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _save_tiles(self, ix, iy, das, stream_cube_id=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        def save_tile(ix, iy, das, stream_cube_id=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireStreamCompressedZStack(self, i, ix, iy, stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireStreamTile(self, i, ix, iy, stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _getTileDAs(self, i, ix, iy):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _acquireTiles(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_z_on_focus_plane(self, x, y):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_triangulated_focus_point(self, x, y):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _refocus(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _get_zstack_levels(self, focus_value):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _adjustFocus(self, das, i, ix, iy):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _stitchTiles(self, da_list):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateTiledAcquisitionTime(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateTiledAcquisitionMemory(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def acquireTiledArea(streams, stage, area, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def estimateOverviewTime(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def acquireOverview(streams, stage, areas, focus, detector, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        Each area is defined by 4 floats :left, top, right, bottom positions
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def __init__(self, streams, stage, areas, focus, detector, future=None,
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def estimate_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def _pass_future_progress(self, future, start: float, end: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_stream_based_bbox(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:        # fall back to a small fov (default)
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_tiled_bboxes(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_fov_based_bbox(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    :param fov: the fov to acquire, default is DEFAULT_FOV
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_fov(s: Stream):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def clip_tiling_bbox_to_range(
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:def get_zstack_levels(zsteps: int, zstep_size: float, rel: bool = False, focuser=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_tiledacq.py:    :param rel: If this is False (default), then z stack levels are in absolute values. If rel is set to True then
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _insert_tile_to_grid(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _find_closest_tile(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _updateGrid(self, direction):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _estimateROI(self, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _estimateMatch(self, imageA, imageB, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _get_shift(self, prev_tile, tile, shift):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _register_horizontally(self, row, col, xdir):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _register_vertically(self, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _compute_registration(self, tile, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def addTile(self, tile, dependent_tiles=None):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def getPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _insert_tile_to_grid(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _get_shift(self, prev_tile, tile):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _compute_registration(self, tile, row, col):
/home/meteor/development/odemis/src/odemis/acq/stitching/_registrar.py:    def _assemble_mosaic(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _check_square_pixel(self, st):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_roi_rep_pxs_links(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_camera_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_histogram(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_hwvas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_weakref(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_default_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        Check the default values for the FluoStream are fitting the current HW
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_sem(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_hwvas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_live_conf(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_conf_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_progressive_future(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_sync_future_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_ar(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_fuz(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_mn(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_count(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_only(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_sstage_all(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_scan_stage_wrapper(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_scan_stage_wrapper_noccd(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_roi_out_of_stage_limits(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_done(self, _):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def on_progress_update(self, _, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_live_stream(self):  # TODO  this one has still exposureTime
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streakcam_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_gui_vas(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_stream_va_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_live_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_streak_acq_integrated_images_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_cl_se_ebic(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _roiToPhys(self, repst):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_arpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_arpol_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_stream_va_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_acq_integrated_images(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_acq_integrated_images_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spec_light_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_spec_light(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ebic_live_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ebic_acq(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_acq_live_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spec_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_repetitions_and_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cancel_active(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mnchr_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cl_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_fluo(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_cl(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_small_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_int_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_uint16_hist(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_uint16_hist_white(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_ar_data(self, shape, tweak=0, pol=None):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_das(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_allpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpol_1pol(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_arpolarimetry(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_ar_large_image(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def on_im(im):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_spectrum_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_das(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_2d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_0d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_1d(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_spectrum_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_temporal_spectrum_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_temporal_spectrum_false_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def _create_chronograph_data(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_chronograph(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_chronograph_calib_bg(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_temporal_spectrum(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_mean_chronograph(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_tiled_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream_pan(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def getTileMock(self, x, y, zoom):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_tiled_stream_zoom(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:        def getTileMock(self, x, y, zoom):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_rgb_updatable_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:    def test_pixel_coordinates(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_test.py:def roi_to_phys(repst):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_spot_alignment(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_spot_alignment_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def on_progress_update(self, future, past, left):
/home/meteor/development/odemis/src/odemis/acq/test/spot_alignment_test.py:    def test_aligned_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:By default, it listens on 127.0.0.1:6000 for a connection. Just launch the EXE with default settings
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def _on_image(self, im):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def validate_scan_sim(self, das, repetition, dwelltime, pixel_size):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_flim_acq_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def _validate_rep(self, s):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_repetitions_and_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_cancelling(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_rep(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_small_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_setting_stream_tcd_live(self):
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:        # TODO: by default the simulator only sends data for 5s... every 1.5s => 3 data
/home/meteor/development/odemis/src/odemis/acq/test/flim_test.py:    def test_rep_ss(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_entries(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_get_streams_settings(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_set_streams_settings(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_set_streams_settings_missing_keys(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:        self.assertEqual(s.detExposureTime.value, 0.89)  # As defined in the "Few settings" entry
/home/meteor/development/odemis/src/odemis/acq/test/stream_settings_test.py:    def test_get_settings_order(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_set_roi(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_next_rectangle(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_complete(self):
/home/meteor/development/odemis/src/odemis/acq/test/leech_test.py:    def test_shape_1(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def __init__(self, cnvs=None):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def check_point_proximity(self, v_point):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def draw(self, ctx, shift=(0, 0), scale=1.0):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def copy(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def move_to(self, pos):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def get_state(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def restore_state(self, state):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def to_dict(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def from_dict(shape, tab_data):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def set_rotation(self, target_rotation):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def reset(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_overview_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_estimateTiledAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_overview_acquisition_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def on_acquisition_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_poly_field_indices_overlap_small_roa(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_acquire_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_coverage_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_progress_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_cancel_ROA(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_movement_rotation_correction(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        f = fastem.acquire(roa, path_storage, "default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement_overlap(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_abs_stage_movement_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_stage_pos_outside_roa(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_pre_calibrate(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_get_pos_first_tile(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                          self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calibration_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_save_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calculate_beam_shift_cor_indices(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path=None, username="default", pre_calibrations=None,
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_save_full_cells(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:        def _image_received(*args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:                                      self.ebeam_focus, roa, path="test-path", username="default",
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_pre_calibrate(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_test.py:    def test_calibration_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_align_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_cancel_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_current_aligner_position(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def check_move_aligner_to_target(self, target):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_current_position(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_smaract_stage_fallback_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_sem_imaging_area_after_loading_1st_method(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the sem imaging area, and grid1 will be chosen by default.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the fm imaging area, and grid1 will be chosen by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_sem_imaging_area_after_loading_2nd_method(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to grid1, and sem imaging area will be chosen by default.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid1_in_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # move the stage to the fm imaging area, and grid1 will be chosen by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_grid2_in_sem_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid1_to_grid2_in_sem_imaging_area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid2_to_grid1_in_sem_imaging_area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_sem_to_fm(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid1_to_grid2_in_fm_imaging_Area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # now the grid is grid1 by default
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_from_grid2_to_grid1_in_fm_imaging_Area(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_to_sem_from_fm(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_transformFromSEMToMeteor(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # the grid positions in the metadata of the stage component are defined without rz axes.
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_log_linked_stage_positions(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_switching_movements(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def _test_3d_transformations(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_to_posture(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_stage_movement(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_transformation_calculation(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_scan_rotation(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_component_metadata_update(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_switching_consistency(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_moving_in_grid1_fm_imaging_area_after_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown_label_at_initialization(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_sample_switch_procedures(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_cancel_loading(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_unknown(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes_but_without_difference(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_linear_axes_but_without_common_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_only_rotation_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_rotation_axes_no_difference(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_rotation_axes_missing_axis(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_no_common_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_lin_rot_axes(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_progress(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_get_progress_lin_rot(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:    def test_isNearPosition(self):
/home/meteor/development/odemis/src/odemis/acq/test/move_test.py:        # test user defined tolerance
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_overview(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_live(self):
/home/meteor/development/odemis/src/odemis/acq/test/fastem_conf_test.py:    def test_configure_scanner_megafield(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def test_feature_acquisitions(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/feature_acq_test.py:    def test_feature_posture_positions(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_encoder(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_decoder(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_save_read_features(self):
/home/meteor/development/odemis/src/odemis/acq/test/feature_test.py:    def test_feature_milling_tasks(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_multi(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_full(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_simple_arpol(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_background(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_compensation(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_load_full(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_compensate(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_compensate_out(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_angular_spec_compensation(self):
/home/meteor/development/odemis/src/odemis/acq/test/calibration_test.py:    def test_write_read_trigger_delays(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def test_overlay_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def test_acq_fine_align(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_overlay_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def __init__(self, name):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def get(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_getSingleFrame(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_single_frame_acquisition_VA(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_simple(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_metadata(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_sync_sem_ccd(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_sync_path_guess(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_leech(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def on_progress_update(self, future, start, end):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_valid_folding(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_folding_same_detector_name(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_folding_pairs_of_two(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def _on_progress_update(self, f, s, e):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_only_FM_streams_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_only_SEM_streams_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_FM_and_SEM_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/acq_test.py:    def test_settings_observer_metadata_with_zstack(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:def path_pos_to_phys_pos(pos, comp, axis):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    Convert a generic position definition into an actual position (as reported in comp.position)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    pos (float, str, or tuple): a generic definition of the position
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    axis_def = comp.axes[axis]
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    if hasattr(axis_def, "choices"):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        for key, value in axis_def.choices.items():
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:def assert_pos_as_in_mode(t: unittest.TestCase, comp, mode):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    Check the position of the given component is as defined for the
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    specified mode (for all the axes defined in the specified mode)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        axis_def = comp.axes[axis]
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:            choices = axis_def.choices
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_queue(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_wrong_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_spec_switch_mirror(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        # set the spec-switch mirror to the default retracted position (FAV_POS_DEACTIVE)
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:        # exclude the check for pos in mode for the spec-switch, it's not in light-in-align mode by default
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_acq_quality(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def assert_pos_align(self, expected_pos_name: str):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_guess_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_mode(self):
/home/meteor/development/odemis/src/odemis/acq/test/path_test.py:    def test_set_path_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def test_drift_stream(self):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/test/stream_drift_test.py:    def on_progress_update(self, future, past, left):
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/test/example_drifted.h5 matches
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_acquire(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimate(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_updateScannerSettings(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:        # Check in the order defined in _updateSEMSettings
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_estimateCorrectionPeriod(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/drift/test/drift_test.py:    def test_identical_inputs(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/test/example_input.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/__pycache__/__init__.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/drift/__pycache__/__init__.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def __init__(self, scanner, detector, region, dwell_time, max_pixels=MAX_PIXELS, follow_drift=True):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:        logging.info("Anchor region defined with scale=%s, res=%s, trans=%s",
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimate(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def estimateCorrectionPeriod(period, t, repetitions):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:    def _updateScannerSettings(self):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:def GuessAnchorRegion(whole_img, sample_region):
/home/meteor/development/odemis/src/odemis/acq/drift/__init__.py:def align_reference_image(
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:# to identify a ROI which must still be defined by the user
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def __init__(self, name, detector, dataflow, emitter, focuser=None, opm=None,
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:          at initialisation. By default, it will contain no data.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # Don't call at init, so don't set metadata if default value
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _init_thread(self, period=0.1):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def emitter(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def detector(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def focuser(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def det_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def emt_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def axis_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def __str__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getVAs(self, comp, va_names):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateVAs(self, comp, prefix, va_names):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _va_sync_setter(self, origva, v):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _va_sync_from_hw(self, lva, v):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateAxis(self, axis_name, actuator):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        axis_name (str): the name of the axis to define
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateAxes(self, axis_map):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _duplicateVA(self, va, setter=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _index_in_va_order(self, va_entry):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getEmitterVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getDetectorVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _linkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        Link the axes, which are defined as local VA's,
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:                moves.setdefault(actuator, {})[axis_name] = pos
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onAxisMoveDone(self, f):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _update_linked_position(self, act, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _update_linked_axis(self, va_name, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _unlinkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # This default implementation returns the shortest possible time, taking
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setStatus(self, level, message=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateDRange(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _guessDRangeFromDetector(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _getDisplayIRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            # default to small value so that it easily fits in the FoV.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _shouldUpdateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _image_thread(wstream, period=0.1):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onAutoBC(self, enabled):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onOutliers(self, outliers):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _recomputeIntensityRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _setTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        data (DataArray): the raw data to use, default to .raw[0] - background
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getPixelCoordinates(self, p_pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:        # uses a different definition of shear.
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getBoundingBox(self, im=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            ValueError: If the stream has no (spatial) data and stream's image is not defined
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:            raise ValueError("Cannot compute bounding-box as stream has no data and stream's image is not defined")
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def getRawMetadata(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_base.py:    def force_image_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, forcemd=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def getSingleFrame(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        Overwritten by children if they have a dedicated method to get a single frame otherwise the default
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            def on_new_data(future):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _shouldUpdateHistogram(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            logging.debug("Cannot find a scale defined for the %s, using (1, 1) instead" % self.name)
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        Doesn't do anything if no writable resolution/translation VA's are defined on the scanner.
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onDwellTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onResolution(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            logging.debug("Cannot find a scale defined for the %s, using (1, 1) instead" % self.name)
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onMove(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _DoPrepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:#     def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_pxsize(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def prepare(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, blanker=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _prepare_opm(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _startSpot(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _stopSpot(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, emtvas=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _stop_light(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # the stream, so use the definition: pxs = sensor pxs * binning / mag
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, emtvas=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _getCount(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, em_filter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # be selected. The difficulty comes to pick the default value. We try
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # default value. In that case, we pick the emission value that matches
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def onExcitation(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def onEmission(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _get_current_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_emission(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setup_excitation(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _protect_detector(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_streak_mode(self, streak: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _on_mcp_gain(self, gain: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:        # We use resolution as a shortcut to define the pitch between pixels
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onZoom(self, z):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onResolution(self, r):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _setResolution(self, res):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, dataflow, emitter, scanner, em_filter,
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def scanner(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def setting_stream(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _getScannerVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def __init__(self, name, detector, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _recomputeIntensityRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:            # TODO: allows to define the power via a VA on the stream
/home/meteor/development/odemis/src/odemis/acq/stream/_live.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _shouldUpdateHistogram(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _clean_raw(self, raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                    logging.warning(u"Metadata for 3D data invalid. Using default pixel size 10µm")
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                    logging.warning(u"Metadata for 3D data invalid. Using default centre position 0")
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _on_zIndex(self, val):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _on_max_projection(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onTint(self, tint):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        default_tint = (0, 255, 0)  # green is most typical
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:                default_tint = conversion.wavelength2rgb(numpy.mean(em_range))
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        tint = raw.metadata.get(model.MD_USER_TINT, default_tint)
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            self.tint.value = default_tint
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, data, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            def dis_bbtl(v):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setBackground(self, bg_data):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, image, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        default_dims = "CTZYX"
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:            default_dims = "CAZYX"
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:        dims = image.metadata.get(model.MD_DIMS, default_dims[-image.ndim::])
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_projection_vas(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _init_thread(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateDRange(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setAngle(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setWavelength(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onTimeSelect(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onWavelengthSelect(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setLine(self, line):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _get_bandwidth_in_pixel(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _updateCalibratedData(self, data=None, bckg=None, coef=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setBackground(self, bckg):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _setEffComp(self, coef):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _force_selected_spectrum_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onCalib(self, unused):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onSelectionWidth(self, width):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def onSpectrumBandwidth(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def __init__(self, name, raw, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def _clean_raw(self, raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_static.py:    def update(self, raw):
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_sync.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_static.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_sync.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/__init__.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_helper.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_live.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_base.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_projection.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/__init__.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_base.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_helper.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_live.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_projection.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/stream/__pycache__/_static.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:          The first stream with .repetition will be used to define
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:          The first stream with .useScanStage will be used to define a scanning
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    #     def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def streams(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def leeches(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateProgress(self, future, dur, current, tot, bonus=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _restoreHardwareSettings(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getPixelSize(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getSpotPositions(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getLeftTopPositionPhys(self) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getScanStagePositions(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onHwSyncData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _preprocessData(self, n, data, i):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _get_center_pxs(self, rep: Tuple[int, int],
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assemble2DData(self, rep, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleTiles(self, rep, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleAnchorData(self, data_list):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _startLeeches(self, img_time, tot_num, shape):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _stopLeeches(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:         :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveDataTiles(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData2D(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _supports_hw_sync(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettingsHwSync(self) -> Tuple[float, int]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionHwSyncEbeam(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:            # Configure the CCD to the defined exposure time, and hardware sync + get frame duration
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionEbeam(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _waitForImage(self, img_time):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, n, px_idx, img_time, sem_time,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettingsScanStage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisitionScanStage(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:                        # MD_POS default to the center of the sample stage, but it needs to be the position
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _get_next_rectangle(self, rep: Tuple[int, int], acq_num: int, px_time: float,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n, raw_data, px_idx, px_pos, rep, pol_idx=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:         :param pol_idx: polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, x, y, img_time):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _getNumDriftCors(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n, raw_data, px_idx, px_pos, rep, pol_idx=0):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:            # for long exposure times. Should be defined on the streak-ccd.metadata[MD_CALIB]["intensity_limit"]
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _acquireImage(self, n, px_idx, img_time, sem_time,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        # overrides the default _acquireImage to check the light intensity after every image from the
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _check_light_intensity(self, raw_data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleLiveData(self, n: int, raw_data: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        :param pol_idx: (int) polarisation index related to name as defined in pos_polarizations variable
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        # MD_POS default to position at the center of the FoV, but it needs to be
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _assembleFinalData(self, n, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, streams):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _estimateRawAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onCompletedData(self, n, raw_das):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _adjustHardwareSettings(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onData(self, n, df, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def __init__(self, name, stream, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:        It acquires by scanning the defined ROI multiple times. Each scanned
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def streams(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _prepareHardware(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _computeROISettings(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _runAcquisition(self, future) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onAcqStop(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _add_frame(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _updateProgress(self, future, dur, current, tot, bonus=2):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def _cancelAcquisition(self, future):
/home/meteor/development/odemis/src/odemis/acq/stream/_sync.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, scanner=None, sstage=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:            # If not integration time, default is 1 image.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # We overwrite the VA provided by LiveStream to define a setter.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateHistogram(self, data=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setIntegrationTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkIntTime2HwExpTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkIntTime2HwExpTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onIntegrationTime(self, intTime):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _intTime2NumImages(self, intTime):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def scanner(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _getScannerVA(self, vaname):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onMagnification(self, mag):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _adaptROI(self, roi, rep, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _fitROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _computePixelSize(self, roi, rep):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateROIAndPixelSize(self, roi, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # If ROI is undefined => link rep and pxs as if the ROI was full
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setPixelSize(self, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setRepetition(self, repetition):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # If ROI is undefined => link repetition and pxs as if ROI is full
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _getPixelSizeRange(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def guessFoV(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, light=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPower(self, power: float):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setup_light(self, ch_power: Optional[float] = None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _stop_light(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, streak_unit, streak_delay,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.pixelSize.value *= 30  # increase default value to decrease default repetition rate
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _is_active_setter(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _suspend(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _protect_detector(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_streak_mode(self, streak: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_mcp_gain(self, gain: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _on_center_wavelength(self, wl: float) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, analyzer=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwAxes(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPolarization(self, pol):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPolarizationMove(self, f):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # (due to the long exposure time). So make the default pixel size bigger.
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, spectrometer, spectrograph, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.pixelSize.value *= 30  # increase default value to decrease default repetition rate
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _unlinkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onBinning(self, _=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onMagnification(self, mag):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # For the live view, we need a way to define the scale and resolution,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        # current SEM pixelSize/mag/FoV) to define the scale. The ROI is always
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _applyROI(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onPixelSize(self, pxs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onDwellTime(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onResolution(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, emt_dataflow, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateAcquisitionTime(self) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewEmitterData(self, dataflow: model.DataFlow, data: model.DataArray) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateMD(self, md: Dict[str, Any]) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow: model.DataFlow, data: model.DataArray) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _restart_acquisition(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active: bool) -> None:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _startAcquisition(self, future=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _set_dwell_time(self, dt: float) -> float:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _set_hw_dwell_time(self, dt: float) -> float:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _linkHwVAs(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onResolution(self, value: Tuple[int, int]):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, ccd, emitter, emd, opm=None):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        self.repetition = model.ResolutionVA((4, 4),  # good default
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def acquire(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _result_wrapper(self, f) -> Callable:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        def result_as_da(timeout=None) -> Tuple[List[model.DataArray], Optional[Exception]]:
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, emitter, scanner, time_correlator,
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:        A helper stream used to define FLIM acquisition settings and run a live setting stream
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def estimateAcquisitionTime(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _append(self, count, date):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _setPower(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onActive(self, active):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def __init__(self, name, detector, dataflow, emitter, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _histogram_thread(wstream):
/home/meteor/development/odemis/src/odemis/acq/stream/_helper.py:    def _onNewData(self, dataflow, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _image_thread(wprojection):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _shouldUpdateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:        # Don't call at init, so don't set metadata if default value
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onIntensityRange(self, irange):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _shouldUpdateImageEntirely(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def onTint(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def force_image_update(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPoint(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getBackground(self, pol_mode):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _processBackground(self, data, pol_mode, clip_data=True):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _resizeImage(self, data, size):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2Polar(self, ebeam_pos, pol_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            polar_data = polar_cache.setdefault(ebeam_pos, {})[pol_pos]
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # define the size of the image for polar representation in GUI
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPolarization(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:            output_size = (90, 360)  # Note: increase if data is high def
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsVis(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getRawData(self, ebeam_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectAsRaw(self, ebeam_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                output_size = (400, 600)  # defines the resolution of the displayed image
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                    # Correct image for background. It must match the polarization (defaulting to MD_POL_NONE).
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _project2RGBPolar(self, ebeam_pos, pol_pos, cache_raw):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:                # define the size of the image for polar representation in GUI
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onPolarimetry(self, pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onBackground(self, data):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsVis(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __new__(cls, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _update_rect(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onMpp(self, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onRect(self, rect):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _set_mpp(self, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectXY2RGB(self, data, tint=(255, 255, 255)):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getPixelCoordinates(self, p_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _onZIndex(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_max_projection(self, value):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getBoundingBox(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _zFromMpp(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _rectWorldToPixel(self, rect):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getTile(
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _projectTile(self, tile):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _getTilesFromSelectedArea(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:        Get the tiles inside the region defined by .rect and .mpp
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_tint(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_spectrumBandwidth(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def getRawValue(self, pixel_pos):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_line(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_time(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_angle(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selection_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selection_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _find_metadata(self, md):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_time(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_angle(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_wl(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_new_spec_data(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_pixel(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_width(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _on_selected_wl(self, _):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _computeSpec(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def projectAsRaw(self):
/home/meteor/development/odemis/src/odemis/acq/stream/_projection.py:    def _updateImage(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __init__(self, operator=None, streams=None, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:            By default operator is an average function.
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __str__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __len__(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __getitem__(self, index):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def __contains__(self, sp):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def add_stream(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def remove_stream(self, stream):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def on_stream_update_changed(self, _=None):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def getProjections(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:#     def getImage(self, rect, mpp):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:#         #  that will define where to save the result
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def getImages(self):
/home/meteor/development/odemis/src/odemis/acq/stream/__init__.py:    def get_projections_by_type(self, stream_types):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def acquire(streams, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def acquireZStack(streams, zlevels, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, future, streams, zlevels, settings_obs):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def cancel(self, _):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def estimate_total_duration(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def estimateZStackAcquisitionTime(streams, zlevels):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def estimateTime(streams):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def foldStreams(streams, reuse=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def computeThumbnail(streamTree, acqTask):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def _weight_stream(stream):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:def sortStreams(streams):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, streams, future, settings_obs=None):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def _adjust_metadata(self, raw_data):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def _on_progress_update(self, f, start, end):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def __init__(self, microscope: model.HwComponent, components: Set[model.HwComponent]):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    # For the .position, the axes definition may contain also the "user-friendly" name
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    def update_settings(value: Dict[str, float], comp_name=comp.name, va_name=va_name, comp=comp):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                            axes_def: Dict[str, model.Axis] = comp.axes
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                if (axis_name in axes_def and hasattr(axes_def[axis_name], "choices")
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    and isinstance(axes_def[axis_name].choices, dict)
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    and p in axes_def[axis_name].choices
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                                    value[axis_name] = f"{p} ({axes_def[axis_name].choices[p]})"
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:                    def update_settings(value, comp_name=comp.name, va_name=va_name):
/home/meteor/development/odemis/src/odemis/acq/acqmng.py:    def get_all_settings(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/fastem.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/leech.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/calibration.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/millmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/move.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/calibration.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/leech.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/acqmng.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/acqmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/__pycache__/move.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:def align(scanner, multibeam, descanner, detector, stage, ccd, beamshift, det_rotator,
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:def estimate_calibration_time(calibrations):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def __init__(self, future, scanner, multibeam, descanner, detector, stage, ccd, beamshift, det_rotator,
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:        :param beamshift: (tfsbc.BeamShiftController) Component that controls the beamshift deflection.
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:                # def _pass_future_progress(sub_f, start, end):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def run_calibration(self, calibration, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def cancel(self, future):
/home/meteor/development/odemis/src/odemis/acq/align/fastem.py:    def estimate_calibration_time(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_image_pair(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_synthetic_images(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/keypoint_test.py:    def test_no_match(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/blank_image.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_cancel_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/roi_autofocus_test.py:    def test_estimate_autofocus_in_roi(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_noise(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_missing_point(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_cosmic_ray(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_divide_and_find_center_grid_noise_missing_point_cosmic_ray(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_single_element(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_distorted_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output_missing_point_3x3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled__distorted_10x10(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_transformation_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_shuffled_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/coordinates_test.py:    def test_precomputed_output_missing_point_40x40(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_opt(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_focus_at_boundaries(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_different_starting_positions(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical_multiple_runs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/clspots_optical_autofocus_test.py:    def test_autofocus_optical_start_at_good_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def test_convert_das_to_numpy_stack(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/tdct_test.py:    def test_run_tdct_correlation(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_cosmic_ray.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_optical_autofocus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_image_translation_prealign(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_image_dark_gain(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_progress(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def on_done(self, future):
/home/meteor/development/odemis/src/odemis/acq/align/test/fastem_test.py:    def on_progress_update(self, future, start, end):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/navcam-calib2.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_off_on_spccd(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_off_on_ccd(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_1pix_off_on(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_timeout(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_slit_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_light_already_on(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/light_test.py:    def test_no_detection_of_change(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/transform_test.py:    def test_calculate_transform(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/brightlight-off-slit-ccd.ome.tiff matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/single_part.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/sem_hole.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_spot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_center_spot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def __init__(self, fake_img, aligner):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def _simulate_image(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_close_to_image_edge(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_affine(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_similarity(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_with_shear(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_with_rotation(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_wrong_method(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/spot_test.py:    def test_find_grid_on_image(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def _prepare_hardware(self, ebeam_kwargs=None, ebeam_mag=2000, ccd_img=None):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay_failure(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_find_overlay_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_bad_mag(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_ratio_4_3(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/find_overlay_test.py:    def test_ratio_3_4(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/centers_grid.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/real_optical.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_missing_point.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def _move_to_vacuum(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_hole_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_sh_hole_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_find_lens_center(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_no_hole(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_hole_detection(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_update_offset_rot(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_align_offset(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_rotation_calculation(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_hfw_shift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:    def test_res_shift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/delphi_test.py:#     def test_scan_pattern(self):
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/one_spot.h5 matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/test/grid_10x10.h5 matches
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_measure_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_opt(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_sem(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_sem_hint(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_det_external(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_spectrograph(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_ded_spectrograph(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_one_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_multi_det(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_spect(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/autofocus_test.py:    def test_autofocus_slit(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_determine_z_position(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_key_error(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    # def test_module_not_found_error(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_pos_shifted(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_cancel(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/z_localization_test.py:    def test_measure_z_bad_angle(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:        # rng = numpy.random.default_rng(0)
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_identical_inputs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_known_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_random_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_different_precisions(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_identical_inputs_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_known_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_random_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_different_precisions_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_identical_inputs(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_random_drift(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_identical_inputs_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/test/shift_test.py:    def test_small_random_drift_noisy(self):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def do_autofocus_in_roi(
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def estimate_autofocus_in_roi_time(n_focus_points, detector, focus, focus_rng, average_focus_time=None):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def _cancel_autofocus_bbox(future):
/home/meteor/development/odemis/src/odemis/acq/align/roi_autofocus.py:def autofocus_in_roi(
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/fastem.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/autofocus.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/shift.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/find_overlay.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/shift.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/autofocus.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/find_overlay.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/align/__pycache__/z_localization.cpython-38.pyc matches
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:    # Sift is not installed by default, check first if it's available
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:# Missing defines from OpenCV
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:def FindTransform(ima, imb, fd_type=None):
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:            search_params = dict(checks=32)  # default value
/home/meteor/development/odemis/src/odemis/acq/align/keypoint.py:def preprocess(im, invert, flip, crop, gaussian_sigma, eqhis):
/home/meteor/development/odemis/src/odemis/acq/align/transform.py:def CalculateTransform(optical_coordinates, electron_coordinates, skew=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def FindOverlay(repetitions, dwell_time, max_allowed_diff, escan, ccd, detector, skew=False, bgsub=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _DoFindOverlay(future, repetitions, dwell_time, max_allowed_diff, escan,
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:                def dist_to_p1(p):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _CancelFindOverlay(future):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _computeGridRatio(coord, shape):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def estimateOverlayTime(dwell_time, repetitions):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _transformMetadata(optical_image, transformation_values, escan, ccd, skew=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _MakeReport(msg, data, optical_image=None, subimages=None):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:def _set_blanker(escan, active):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def __init__(self, repetitions, dwell_time, escan, ccd, detector, bgsub=False):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _save_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _restore_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _discard_data(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _onCCDImage(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _onSpotImage(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _doSpotAcquisition(self, electron_coordinates, scale):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def _doWholeAcquisition(self, electron_coordinates, scale):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def DoAcquisition(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def CancelAcquisition(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def get_sem_fov(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def get_ccd_fov(self):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def sem_roi_to_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:        Converts a ROI defined in the SEM referential a ratio of FoV to a ROI
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:        logging.info("ROI defined at %s m", phys_rect)
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def convert_roi_ratio_to_phys(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def convert_roi_phys_to_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/find_overlay.py:    def configure_ccd(self, roi):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def huang(z, calibration_data):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def thunderstorm(z, calibration_data):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:    This formula is based on the default algorithm used in the Thunderstorm ImageJ plugin as described in the reference
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def solve_psf(z, obs_x, obs_y, calibration_data, model_function=huang):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def determine_z_position(image, calibration_data, fit_tol=0.1):
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:                        4 = Outputted Z position is outside the defined maximum range from the calibration, output is inaccurate
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def measure_z(stigmator, angle: float, pos: Tuple[float, float], stream, logpath: Optional[str]=None) -> ProgressiveFuture:
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def estimate_measure_z(stigmator, angle: float, pos: Tuple[float, float], stream) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def _do_measure_z(f: ProgressiveFuture, stigmator, angle: float, pos: Tuple[float, float], stream,
/home/meteor/development/odemis/src/odemis/acq/align/z_localization.py:def _cancel_localization(future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def getNextImage(det, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def receive_one_image(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AcquireNoBackground(det, dfbkg=None, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoBinaryFocus(future, detector, emt, focus, dfbkg, good_focus, rng_focus):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:                # so let's "center" on it. That's better than the default behaviour
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoExhaustiveFocus(future, detector, emt, focus, dfbkg, good_focus, rng_focus):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelAutoFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def estimateAcquisitionTime(detector, scanner=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def estimateAutoFocusTime(detector, emt, focus: model.Actuator, dfbkg=None, good_focus=None, rng_focus=None, method=MTD_BINARY) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def Sparc2AutoFocus(align_mode, opm, streams=None, start_autofocus=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    This automatically defines the spectrograph to use and the detectors.
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _cancelSparc2ManualFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def Sparc2ManualFocus(opm, align_mode, toggled=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoSparc2ManualFocus(opm, spgr: model.Actuator, bl: model.Emitter, align_mode: str, toggled: bool = True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def GetSpectrometerFocusingDetectors(focuser: model.Actuator) -> List[model.Detector]:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _findSparc2Focuser(align_mode: str) -> model.Actuator:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _getSpectrometerFocusingComponents(focuser: model.Actuator) -> Tuple[
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _findSameAffects(roles: List[str], affected: List[str]) -> model.Component:
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoSparc2AutoFocus(future, streams, align_mode, opm, dets, spgr, selector, bl, focuser, start_autofocus=True):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def updateProgress(subf, start, end):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelSparc2AutoFocus(future):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AutoFocus(detector, emt, focus, dfbkg=None, good_focus=None, rng_focus=None, method=MTD_BINARY):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def AutoFocusSpectrometer(spectrograph, focuser, detectors, selector=None, streams=None):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _totalAutoFocusTime(spectrograph, focuser, detectors, selector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _updateAFSProgress(future, af_dur, grating_moves, detector_moves):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def CLSpotsAutoFocus(detector, focus, good_focus=None, rng_focus=None, method=MTD_EXHAUSTIVE):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _mapDetectorToSelector(selector, detectors):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _playStream(detector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _DoAutoFocusSpectrometer(future, spectrograph, focuser, detectors, selector, streams):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:    def is_current_det(d):
/home/meteor/development/odemis/src/odemis/acq/align/autofocus.py:def _CancelAutoFocusSpectrometer(future):
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def _convert_das_to_numpy_stack(das: List[model.DataArray]) -> numpy.ndarray:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def get_optimized_z_gauss(das: List[model.DataArray], x: int, y: int, z: int, show: bool = False) -> float:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def run_tdct_correlation(fib_coords: numpy.ndarray,
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def get_reprojected_poi_coordinate(correlation_results: dict) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/align/tdct.py:def parse_3dct_yaml_file(path: str) -> Tuple[float, float]:
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def list_hw_settings(escan, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def restore_hw_settings(escan, ccd, hw_settings):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def DelphiCalibration(main_data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoDelphiCalibration(future, main_data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _StoreConfig(path, shid, calib_values):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelDelphiCalibration(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateDelphiCalibration():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def AlignAndOffset(ccd, detector, escan, sem_stage, opt_stage, focus, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoAlignAndOffset(future, ccd, detector, escan, sem_stage, opt_stage, focus, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelAlignAndOffset(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateOffsetTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def RotationAndScaling(ccd, detector, escan, sem_stage, opt_stage, focus, offset,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoRotationAndScaling(future, ccd, detector, escan, sem_stage, opt_stage, focus,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:    def find_spot(n):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelRotationAndScaling(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateRotationAndScalingTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def HoleDetection(detector, escan, sem_stage, ebeam_focus, manual=False,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoHoleDetection(future, detector, escan, sem_stage, ebeam_focus, manual=False,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:        def find_sh_hole(holep):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:        def find_focus():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelHoleDetection(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateHoleDetectionTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def FindCircleCenter(image, radius, max_diff, darkest=False):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def FindRingCenter(image):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def UpdateOffsetAndRotation(new_first_hole, new_second_hole, expected_first_hole,
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def LensAlignment(navcam, sem_stage, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoLensAlignment(future, navcam, sem_stage, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateLensAlignmentTime():
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def HFWShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _CancelFuture(future):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoHFWShiftFactor(future, detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateHFWShiftFactorTime(et):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def ResolutionShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoResolutionShiftFactor(future, detector, escan, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def estimateResolutionShiftFactorTime(et):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def ScaleShiftFactor(detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi.py:def _DoScaleShiftFactor(future, detector, escan, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def turnOnLight(bl, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def _cancelTurnOnLight(f):
/home/meteor/development/odemis/src/odemis/acq/align/light.py:def _doTurnOnLight(f, bl, ccd):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def getch():
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def print_col(colour, s, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def input_col(colour, s):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def _discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def __init__(self, sem_stage, opt_focus, ebeam_focus, opt_stepsize, ebeam_stepsize):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def _move_focus(self):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    def focusByArrow(self, rollback_pos=None):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def man_calib(logpath, keep_loaded=False):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                    # the range of the tmcm axes to be defined per sample
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                    def ask_user_to_focus(n):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:def main(args):
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:                        default=0, help="set verbosity level (0-2, default = 0)")
/home/meteor/development/odemis/src/odemis/acq/align/delphi_man_calib.py:    parser.add_argument("--keep-loaded", dest="keep_loaded", action="store_true", default=False,
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def FindEbeamCenter(ccd, detector, escan):
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def _ConvertCoordinates(coord, img):
/home/meteor/development/odemis/src/odemis/acq/align/__init__.py:def discard_data(df, data):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def DivideInNeighborhoods(data, number_of_spots, scale, sensitivity_limit=100):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def ReconstructCoordinates(subimage_coordinates, spot_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def FilterOutliers(image, subimages, subimage_coordinates, expected_spots):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def MatchCoordinates(input_coordinates, electron_coordinates, guess_scale, max_allowed_diff):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _KNNsearch(x_coordinates, y_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _TransformCoordinates(x_coordinates, translation, rotation, scale):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _MatchAndCalculate(transformed_coordinates, optical_coordinates, electron_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:    # Correct for shift if 'too many' points are wrong, with 'too many' defined by:
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _FindOuterOutliers(x_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/coordinates.py:def _FindInnerOutliers(x_coordinates):
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:def _upsampled_dft(data, upsampled_region_size,
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:def MeasureShift(previous_img, current_img, precision=1):
/home/meteor/development/odemis/src/odemis/acq/align/shift.py:    of the maximum of the cross-correlation define the shift vector between the two images.
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def MeasureSNR(image):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def AlignSpot(ccd, stage, escan, focus, type=OBJECTIVE_MOVE, dfbkg=None, rng_f=None, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _DoAlignSpot(future, ccd, stage, escan, focus, type, dfbkg, rng_f, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _CancelAlignSpot(future):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def estimateAlignmentTime(et, dist=None, n_autofocus=2):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _set_blanker(escan, active):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def FindSpot(image, sensitivity_limit=100):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def FindGridSpots(image, repetition, spot_size=18, method=GRID_AFFINE):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:    # Estimate the two most common (orthogonal) directions in the grid of spots, defined in the image coordinate system.
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def CropFoV(ccd, dfbkg=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def CenterSpot(ccd, stage, escan, mx_steps, type=OBJECTIVE_MOVE, dfbkg=None, logpath=None):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _DoCenterSpot(future, ccd, stage, escan, mx_steps, type, dfbkg, logpath):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def _CancelCenterSpot(future):
/home/meteor/development/odemis/src/odemis/acq/align/spot.py:def estimateCenterTime(et, dist=None):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def __init__(self, ccd, detector, escan, opt_stage, focus, pattern):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _save_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _restore_hw_settings(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def _discard_data(self, df, data):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def DoPattern(self):
/home/meteor/development/odemis/src/odemis/acq/align/pattern.py:    def CancelPattern(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:def wrapSimpleStreamIntoFuture(stream):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def __init__(self, stream):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def cancel(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def _run(self):
/home/meteor/development/odemis/src/odemis/acq/_futures.py:    def _image_listener(self, image):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def __init__(self, future: Future, tasks: List[MillingTaskSettings], fib_stream: FIBStream, filename: str = None):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def cancel(self, future: Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def estimate_milling_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run_milling(self, settings: MillingTaskSettings):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:            self.fibsem.set_default_patterning_beam_type(milling_channel)
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def run_milling_tasks(tasks: List[MillingTaskSettings], fib_stream: FIBStream, filename: str = None) -> Future:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def get_associated_tasks(wt: MillingWorkflowTask,
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def __init__(self,
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def cancel(self, future: Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def check_cancelled(self):
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _move_to_milling_position(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _run_milling_tasks(self, feature: CryoFeature, workflow_task: MillingWorkflowTask) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _align_reference_image(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def _acquire_reference_images(self, feature: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:    def get_filename(self, feature: CryoFeature, basename: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/millmng.py:def run_automated_milling(features: List[CryoFeature],
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def create_openfibsem_microscope() -> 'OdemisMicroscope':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    # loads the default config
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_pattern_to_openfibsem(p: MillingPatternParameters) -> 'BasePattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_rectangle_pattern(p: RectanglePatternParameters) -> 'RectanglePattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_trench_pattern(p: TrenchPatternParameters) -> 'TrenchPattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def _convert_microexpansion_pattern(p: MicroexpansionPatternParameters) -> 'MicroExpansionPattern':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_milling_settings(s: MillingSettings) -> 'FibsemMillingSettings':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_task_to_milling_stage(task: MillingTaskSettings) -> 'FibsemMillingStage':
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def convert_milling_tasks_to_milling_stages(milling_tasks: List[MillingTaskSettings]) -> List['FibsemMillingStage']:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def __init__(self, future: futures.Future,
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def cancel(self, future: futures.Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def estimate_milling_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def run_milling(self, stage: 'FibsemMillingStage') -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:    def run(self):
/home/meteor/development/odemis/src/odemis/acq/milling/openfibsem.py:def run_milling_tasks_openfibsem(tasks: List[MillingTaskSettings],
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def tearDownClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_milling_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_milling_task_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/tasks_test.py:    def test_save_load_task_settings(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def setUpClass(cls):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def tearDown(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_automated_milling(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_automated_milling_doesnt_start(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/millmng_test.py:    def test_cancel_workflow(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def setUp(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_assignment(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_dict(self):
/home/meteor/development/odemis/src/odemis/acq/milling/test/patterns_test.py:    def test_generate(self):
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_trench_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_rectangle_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def _draw_microexpansion_pattern(image: model.DataArray,
/home/meteor/development/odemis/src/odemis/acq/milling/plotting.py:def draw_milling_tasks(image: model.DataArray, milling_tasks: Dict[str, MillingTaskSettings]) -> plt.Figure:
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/tasks.cpython-39.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/tasks.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/millmng.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/patterns.cpython-38.pyc matches
Binary file /home/meteor/development/odemis/src/odemis/acq/milling/__pycache__/patterns.cpython-39.pyc matches
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:This module contains structures to define milling tasks and parameters.
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __init__(self, current: float, voltage: float, field_of_view: float, mode: str = "Serial", channel: str = "ion", align: bool = True):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def from_dict(data: dict) -> "MillingSettings":
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __init__(self, milling: dict, patterns: List[MillingPatternParameters], name: str):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def from_dict(data: dict) -> "MillingTaskSettings":
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:def save_milling_tasks(path: str, milling_tasks: Dict[str, MillingTaskSettings]) -> None:
/home/meteor/development/odemis/src/odemis/acq/milling/tasks.py:def load_milling_tasks(path: str) -> Dict[str, MillingTaskSettings]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:This module contains structures to define milling patterns.
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, name: str):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List['MillingPatternParameters']:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, rotation: float = 0.0, center = (0, 0), scan_direction: str = "TopToBottom", name: str = "Rectangle"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'RectanglePatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, spacing: float, center = (0, 0), name: str = "Trench"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'TrenchPatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __init__(self, width: float, height: float, depth: float, spacing: float, center = (0, 0), name: str = "Trench"):
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def to_dict(self) -> dict:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def from_dict(data: dict) -> 'MicroexpansionPatternParameters':
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def __repr__(self) -> str:
/home/meteor/development/odemis/src/odemis/acq/milling/patterns.py:    def generate(self) -> List[MillingPatternParameters]:
/home/meteor/development/odemis/src/odemis/acq/path.py:                 # of picking a grating (non mirror) by default, using a "local axis"
/home/meteor/development/odemis/src/odemis/acq/path.py:            'ek-align': (r"ccd.*",  # Same as "ek", but with grating and filter explicitly set by default
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def submit(self, fn, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/path.py:def affectsGraph(microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __new__(cls, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:            # Consider that by default we are in "fast" acquisition, with the fan
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __del__(self):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _getComponent(self, role):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def setAcqQuality(self, quality):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def setPath(self, mode, detector=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _doSetPath(self, path, detector):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def selectorsToPath(self, target):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _compute_role_to_mode(self, modes: Dict[str, Tuple]) -> Tuple[Tuple[str, str]]:
/home/meteor/development/odemis/src/odemis/acq/path.py:    def guessMode(self, guess_stream):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def getStreamDetector(self, path_stream):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def findNonMirror(self, choices):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def mdToValue(self, comp, md_name):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def affects(self, affecting, affected):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def findPath(self, node1, node2, path=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _setCCDFan(self, enable):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _waitTemperatureReached(self, comp, timeout=None):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/path.py:    def _doSetPath(self, path, detector):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __new__(cls, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:            # Check the version in MD_CALIB, defaults to tfs_1
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def cryoSwitchSamplePosition(self, target: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target_pos: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getMovementProgress(self, current_pos: Dict[str, float], start_pos: Dict[str, float],
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getDistance(self, start: dict, end: dict) -> float:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_stage_metadata(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _cancelCryoMoveSample(self, future):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _run_reference(self, future, component):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _run_sub_move(self, future, component, sub_move):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _update_posture(self, position: Dict[str, float]):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def at_milling_posture(self, pos: Dict[str, float], stage_md: Dict[str, float]) -> bool:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def get_posture_orientation(self, posture: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _initialise_transformation(
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_rot_matrix(self, invert: bool = False) -> RigidTransform:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_sample_from_stage(self, val: List[float], absolute=True) -> List[float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_sample_to_stage(self, val: List[float], absolute=True) -> List[float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _update_conversion(self):
/home/meteor/development/odemis/src/odemis/acq/move.py:        # NOTE: transformations are defined as stage -> sample stage
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_scan_rotation_matrix(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_stage_pos(self, sample_val: Dict[str, float], absolute: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_sample_pos(self, stage_val: Dict[str, float], absolute: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def constrain_stage_pos_axes(self, stage_val: Dict[str, float], fixed_sample_axes: Dict[str, float],
/home/meteor/development/odemis/src/odemis/acq/move.py:        Get the stage (bare) coordinates by constraining the sample plane to a defined plane given by fixed_sample_pos.
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_to_sample_stage_from_stage(self, pos_dep, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _convert_from_sample_stage_to_stage(self, pos, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_pos_vector(self, pos_val, absolute=True):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_movement(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_position(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_sample_stage_from_stage_position(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_movement2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def from_sample_stage_to_stage_position2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_sample_stage_from_stage_position2(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def to_posture(self, pos: Dict[str, float], posture: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_sem_to_milling(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_milling_to_sem(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fm_to_milling(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_milling_to_fm(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:        # this is not the case by default, and needs to be added in the metadata
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:        # defined by the Z axis of the sample plane. Default value is computed by GRID 1 centre of stage (bare) in FM
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _get_fm_sample_z(self, stage_pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float], fix_fm_plane: bool = True) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def create_sample_stage(self):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fib_to_fm(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transform_from_fm_to_fib(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def check_calib_data(self, required_keys: set):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:                # if at loading, and sem is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:                    # if at loading and fm is pressed, choose grid1 by default
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentGridLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromSEMToMeteor(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _transformFromMeteorToSEM(self, pos: Dict[str, float]) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:            # axis with choices defining the position to "parked" and "engaged".
/home/meteor/development/odemis/src/odemis/acq/move.py:                # If not in imaging mode yet, move the stage to the default imaging position
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, microscope):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getTargetPosition(self, target_pos_lbl: int) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def getCurrentPostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getCurrentStagePostureLabel(self, stage_pos: Dict[str, float] = None) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchSamplePosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def get3beamsSafePos(self, active_pos: Dict[str, float], safety_margin: float) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _cryoSwitchAlignPosition(self, target: int):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _doCryoSwitchAlignPosition(self, future, target):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _getCurrentAlignerPositionLabel(self) -> int:
/home/meteor/development/odemis/src/odemis/acq/move.py:        # TODO: should have a POS_ACTIVE_RANGE to define the whole region
/home/meteor/development/odemis/src/odemis/acq/move.py:    def __init__(self, name: str, role: str, stage_bare: model.Actuator , posture_manager: MicroscopePostureManager, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _updatePosition(self, pos_dep):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def _updateSpeed(self, dep_speed):
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveRel(self, shift: Dict[str, float], **kwargs) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveAbs(self, pos: Dict[str, float], **kwargs) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def moveRelChamberReferential(self, shift: Dict[str, float]) -> Future:
/home/meteor/development/odemis/src/odemis/acq/move.py:    def stop(self, axes=None):
/home/meteor/development/odemis/src/odemis/acq/move.py:def calculate_stage_tilt_from_milling_angle(milling_angle: float, pre_tilt: float, column_tilt: int = math.radians(52)) -> float:
/home/meteor/development/odemis/src/odemis/acq/move.py:    :param column_tilt: the column tilt in radians (default TFS = 52deg, Tescan = 55deg)
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_scanner(scanner, mode):
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_detector(detector, roc2, roc3):
/home/meteor/development/odemis/src/odemis/acq/fastem_conf.py:def configure_multibeam(multibeam):
/home/meteor/development/odemis/src/odemis/acq/leech.py:def get_next_rectangle(shape, current, n):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_start(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self, scanner, detector):
/home/meteor/development/odemis/src/odemis/acq/leech.py:        # in seconds, default to "fairly frequent" to work hopefully in most cases
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def tot_drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def max_drift(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def raw(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _setROI(self, roi):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_start(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:            raise ValueError("AnchorDriftCorrector.roi is not defined")
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def series_complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, acq_t, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def __init__(self, detector, selector=None):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def estimateAcquisitionTime(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _get_next_pixels(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def _acquire(self):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def start(self, dt, shape):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def next(self, das):
/home/meteor/development/odemis/src/odemis/acq/leech.py:    def complete(self, das):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(self, name: str,
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def set_posture_position(self, posture: str, position: Dict[str, float]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def get_posture_position(self, posture: str) -> Dict[str, float]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def save_milling_task_data(self,
/home/meteor/development/odemis/src/odemis/acq/feature.py:def get_feature_position_at_posture(pm: MicroscopePostureManager,
/home/meteor/development/odemis/src/odemis/acq/feature.py:def get_features_dict(features: List[CryoFeature]) -> Dict[str, str]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(self, *args, **kwargs):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def object_hook(self, obj):
/home/meteor/development/odemis/src/odemis/acq/feature.py:def save_features(project_dir: str, features: List[CryoFeature]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def read_features(project_dir: str) -> List[CryoFeature]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def load_project_data(path: str) -> dict:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def add_feature_info_to_filename(feature: CryoFeature, filename: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def _create_fibsem_filename(filename: str, acq_type: str) -> str:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def __init__(
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def cancel(self, future: futures.Future) -> bool:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _acquire_feature(self, site: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _run_autofocus(self, site: CryoFeature) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _move_to_site(self, site: CryoFeature):
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _move_focus(self, site: CryoFeature, fm_focus_position: Dict[str, float]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _generate_zlevels(self, zmin: float, zmax: float, zstep: float) -> Dict[Stream, List[float]]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_acquisition_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_movement_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def estimate_total_time(self) -> float:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def run(self) -> Tuple[List[model.DataArray], Exception]:
/home/meteor/development/odemis/src/odemis/acq/feature.py:    def _export_data(self, feature: CryoFeature, data: List[model.DataArray]) -> None:
/home/meteor/development/odemis/src/odemis/acq/feature.py:def acquire_at_features(
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:def get_settings_order(stream):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def __init__(self, file_path, max_entries):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _read(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _write(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def entries(self):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def _get_config_index(self, config_data: list, name: str) -> int:
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def update_data(self, new_entries: List[Dict[str, Any]]):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def update_entries(self, streams: list):
/home/meteor/development/odemis/src/odemis/acq/stream_settings.py:    def apply_settings(self, stream: Stream, name: str):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_ar_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:            ar_data.setdefault(polmode, []).append(da)
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_spectrum_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_temporalspectrum_data(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def get_spectrum_efficiency(das):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def apply_spectrum_corrections(data, bckg=None, coef=None):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def project_angular_spectrum_to_grid_5d(data: model.DataArray) -> model.DataArray:
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def write_trigger_delay_csv(filename, trig_delays):
/home/meteor/development/odemis/src/odemis/acq/calibration.py:def read_trigger_delay_csv(filename, time_choices, trigger_delay_range):
