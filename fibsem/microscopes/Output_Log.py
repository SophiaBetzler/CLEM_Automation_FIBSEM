Component 'AntiBacklash for Filter Wheel':
	role: None
	affects:
	axes (RO Attribute)
		fw:	Axis in -6.949999889344 -> 13.999999778688 rad
	hwVersion (RO Attribute)	value: Unknown
	swVersion (RO Attribute)	value: Unknown (Odemis 3.6.0-246-g9ac466a5b-dirty)
	dependencies (Vigilant Attribute)	 value: {'Optical Actuators'}
	position (RO Vigilant Attribute)	 value: {'fw': 0.08000012496} (unit: m)	{'fw': 4.583669520727194°}
	referenced (RO Vigilant Attribute)	 value: {'fw': True}
	speed (RO Vigilant Attribute)	 value: {'fw': 0.5992109375} (unit: m/s)
	state (RO Vigilant Attribute)	 value: running

Component 'Optical Actuators':
	role: None
	affects:
	axes (RO Attribute)
		fw:	Axis in -6.999999889344 -> 13.999999778688 rad
	hwVersion (RO Attribute)	value: TMCM-6110 (firmware 1.46)
	swVersion (RO Attribute)	value: 3.6.0-246-g9ac466a5b-dirty (serial driver: Unknown)
	dependencies (Vigilant Attribute)	 value: {}
	position (RO Vigilant Attribute)	 value: {'fw': 0.08000012496} (unit: m)	{'fw': 4.583669520727194°}
	referenced (RO Vigilant Attribute)	 value: {'fw': True}
	speed (RO Vigilant Attribute)	 value: {'fw': 0.5992109375} (unit: m/s)
	state (RO Vigilant Attribute)	 value: running

Component 'Optical Objective':
	role: lens
	affects: 'Camera'
	hwVersion (RO Attribute)	value: Optical Objective
	swVersion (RO Attribute)	value: N/A (Odemis 3.6.0-246-g9ac466a5b-dirty)
	dependencies (Vigilant Attribute)	 value: {}
	magnification (Vigilant Attribute)	 value: 84.0 (range: 0.001 → 1000000.0)
	numericalAperture (Vigilant Attribute)	 value: 0.8 (range: 1e-06 → 1000.0)
	refractiveIndex (Vigilant Attribute)	 value: 1.0 (range: 0.01 → 10)
	state (RO Vigilant Attribute)	 value: running

Component 'AntiBacklash Actuators':
	role: None
	affects:
	axes (RO Attribute)
		stig:	Axis in -13.9800032 -> 7.0000016 rad
	hwVersion (RO Attribute)	value: Unknown
	swVersion (RO Attribute)	value: Unknown (Odemis 3.6.0-246-g9ac466a5b-dirty)
	dependencies (Vigilant Attribute)	 value: {'Stigmator Actuator'}
	position (RO Vigilant Attribute)	 value: {'stig': -0.6987680000000001} (unit: m)	{'stig': -40.03645725879751°}
	referenced (RO Vigilant Attribute)	 value: {'stig': True}
	speed (RO Vigilant Attribute)	 value: {'stig': 13.28125} (unit: m/s)
	state (RO Vigilant Attribute)	 value: running

Component 'Light Source':
	role: light
	affects: 'Camera'
	hwVersion (RO Attribute)	value: Omicron LedHUB v1.1801 (s/n 2230228.4) (s/n[1] 2303-126, s/n[3] 2303-138, s/n[4] 2303-41, s/n[6] 2303-40)
	shape (RO Attribute)	value: ()
	swVersion (RO Attribute)	value: 3.6.0-246-g9ac466a5b-dirty (serial driver: ftdi_sio)
	dependencies (Vigilant Attribute)	 value: {}
	power (Vigilant Attribute)	 value: [0.0, 0.0, 0.0, 0.0] (unit: W) (range: (0.0, 0.0, 0.0, 0.0) → (1.4000000000000001, 0.78, 1.45, 0.85))
	spectra (RO Vigilant Attribute)	 value: [(3.75e-07, 3.82e-07, 3.85e-07, 3.8800000000000003e-07, 3.9500000000000003e-07), (4.6000000000000004e-07, 4.6700000000000004e-07, 4.7000000000000005e-07, 4.7300000000000007e-07, 4.800000000000001e-07), (5.450000000000001e-07, 5.520000000000001e-07, 5.550000000000001e-07, 5.580000000000001e-07, 5.650000000000001e-07), (6.15e-07, 6.22e-07, 6.25e-07, 6.280000000000001e-07, 6.350000000000001e-07)] (unit: m)
	state (RO Vigilant Attribute)	 value: running

Component 'Stigmator':
	role: stigmator
	affects: 'Camera'
	axes (RO Attribute)
		rz:	Axis in 0 -> 6.283185 rad
	hwVersion (RO Attribute)	value: Unknown
	swVersion (RO Attribute)	value: Unknown (Odemis 3.6.0-246-g9ac466a5b-dirty)
	dependencies (Vigilant Attribute)	 value: {'AntiBacklash Actuators'}
	position (RO Vigilant Attribute)	 value: {'rz': 6.283185}	{'rz': 359.9999823999061°}
	referenced (RO Vigilant Attribute)	 value: {'rz': True}
	state (RO Vigilant Attribute)	 value: running
	Metadata:
		POS_COR: -0.698768
		CALIB: {0.0262: {'x': {'a': 0.2776726566744459, 'b': 0.005243584563638028, 'c': 522.2281261564915, 'd': 483.0808116203732, 'w0': 9.524395314959095}, 'y': {'a': -0.3617690717345442, 'b': -0.03566609269230697, 'c': 1424.134120732775, 'd': 457.3606790990147, 'w0': 9.388879997702146}, 'feature_angle': -3.1416, 'upsample_factor': 5, 'z_least_confusion': 9.836207241448289e-07, 'z_calibration_range': [-9.836207241448289e-07, 8.616379275855172e-06]}}

Component 'Filter Wheel':
	role: filter
	affects: 'Camera'
	axes (RO Attribute)
		band:	Axis in {0.08: 'pass-through', 0.865398: [5e-07, 5.3e-07], 1.650796: [5.795e-07, 6.105e-07], 2.4361944: [6.63e-07, 7.33e-07]} rad
	hwVersion (RO Attribute)	value: Unknown
	swVersion (RO Attribute)	value: Unknown (Odemis 3.6.0-246-g9ac466a5b-dirty)
	dependencies (Vigilant Attribute)	 value: {'AntiBacklash for Filter Wheel'}
	position (RO Vigilant Attribute)	 value: {'band': 0.08}	{'band': 4.583662361046586°}
	referenced (RO Vigilant Attribute)	 value: {'band': True}
	state (RO Vigilant Attribute)	 value: running

Component 'Optical Focus':
	role: focus
	affects: 'Camera'
	axes (RO Attribute)
		z:	Axis in -0.015 -> 0.015 m
	hwVersion (RO Attribute)	value: SmarAct MCS2-00005114 (s/n MCS2-00005114) with positioners SL...D1SC1
	swVersion (RO Attribute)	value: 1.3.24.106398
	dependencies (Vigilant Attribute)	 value: {}
	position (RO Vigilant Attribute)	 value: {'z': -0.011000060544}
	referenced (RO Vigilant Attribute)	 value: {'z': True}
	speed (RO Vigilant Attribute)	 value: {'z': 0.002} (unit: m/s)
	state (RO Vigilant Attribute)	 value: running
	Metadata:
		FAV_POS_DEACTIVE: {'z': -0.011}
		FAV_POS_ACTIVE: {'z': -0.005}

Component 'METEOR-FIBSEM-Sim':
	role: meteor
	affects:
	hwVersion (RO Attribute)	value: Unknown
	model (RO Attribute)	value: {'METEOR-FIBSEM-Sim': {'class': 'Microscope', 'role': 'meteor'}, 'FIBSEM': {'class': 'autoscript_client.SEM', 'role': 'fibsem', 'init': {'address': 'localhost'}, 'children': {'stage': 'Stage', 'sem-scanner': 'Electron-Beam', 'sem-detector': 'Electron-Detector', 'sem-focus': 'Electron-Focus', 'fib-scanner': 'Ion-Beam', 'fib-detector': 'Ion-Detector', 'fib-focus': 'Ion-Focus'}}, 'Electron-Beam': {'role': 'e-beam', 'init': {'hfw_nomag': 0.25}, 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Electron-Detector': {'role': 'se-detector', 'init': {}, 'properties': {'medianFilter': 3}, 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Electron-Focus': {'role': 'ebeam-focus', 'init': {}, 'affects': ['Electron-Beam'], 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Ion-Beam': {'role': 'ion-beam', 'init': {'hfw_nomag': 0.25}, 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Ion-Detector': {'role': 'se-detector-ion', 'init': {}, 'properties': {'medianFilter': 3}, 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Ion-Focus': {'role': 'ion-focus', 'init': {}, 'affects': ['Ion-Beam'], 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Stage': {'role': 'stage-bare', 'init': {}, 'metadata': {'FAV_POS_DEACTIVE': {'rx': 0.0, 'rz': 1.9076449, 'x': -0.01529, 'y': 0.0506, 'z': 0.01975}, 'SEM_IMAGING_RANGE': {'x': [-0.01, 0.01], 'y': [-0.01, 0.01], 'z': [0.0, 0.005]}, 'FM_IMAGING_RANGE': {'x': [0.039, 0.059], 'y': [-0.01, 0.01], 'z': [0.0, 0.005]}, 'SAMPLE_CENTERS': {'GRID 1': {'x': -0.0026655, 'y': 0.0031458, 'z': 0.0022115}, 'GRID 2': {'x': 0.0033326, 'y': 0.0032286, 'z': 0.0022115}}, 'CALIB': {'version': 'tfs_3', 'dx': 0.048992, 'dy': -0.000288, 'trans-dx': 0.0488, 'trans-dy': 0.0, 'pre-tilt': 0.6108652381980153, 'SEM-Eucentric-Focus': 0.004, 'use_linked_sem_focus_compensation': False, 'use_3d_transforms': True, 'use_scan_rotation': True}, 'FAV_FM_POS_ACTIVE': {'rx': 0.296706, 'rz': 1.5707963268}, 'FAV_SEM_POS_ACTIVE': {'rx': 0.6108, 'rz': 4.7123889804}, 'FAV_MILL_POS_ACTIVE': {'rx': 0.2617993877991494, 'rz': 4.7123889804}}, 'parents': ['FIBSEM'], 'creator': 'FIBSEM'}, 'Light Source': {'class': 'omicronxx.HubxX', 'role': 'light', 'init': {'port': '/dev/ttyFTDI*'}, 'affects': ['Camera']}, 'Optical Objective': {'class': 'static.OpticalLens', 'role': 'lens', 'init': {'mag': 84.0, 'na': 0.8, 'ri': 1.0}, 'affects': ['Camera']}, 'Camera': {'class': 'andorcam3.AndorCam3', 'role': 'ccd', 'init': {'device': 0, 'transp': [-2, 1], 'max_res': [1025, 1275]}, 'properties': {'fanSpeed': 0}, 'metadata': {'ROTATION': -0.099484}}, 'Optical Actuators': {'class': 'tmcm.TMCLController', 'role': None, 'init': {'port': '/dev/ttyTMCM*', 'address': 7, 'param_file': '/usr/share/odemis/meteor-tmcm6110-filterwheel.tmcm.tsv', 'axes': ['fw'], 'ustepsize': [1.227184e-06], 'rng': [[-14, 7]], 'unit': ['rad'], 'refproc': 'Standard', 'refswitch': {'fw': 0}, 'inverted': ['fw']}, 'parents': ['AntiBacklash for Filter Wheel']}, 'AntiBacklash for Filter Wheel': {'class': 'actuator.AntiBacklashActuator', 'role': None, 'init': {'backlash': {'fw': 0.05}}, 'children': {'slave': 'Optical Actuators'}}, 'Filter Wheel': {'class': 'actuator.FixedPositionsActuator', 'role': 'filter', 'dependencies': {'band': 'AntiBacklash for Filter Wheel'}, 'init': {'axis_name': 'fw', 'positions': {0.08: 'pass-through', 0.865398: [5e-07, 5.3e-07], 1.650796: [5.795e-07, 6.105e-07], 2.4361944: [6.63e-07, 7.33e-07]}, 'cycle': 6.283185}, 'affects': ['Camera']}, 'Stigmator Actuator': {'class': 'tmcm.TMCLController', 'role': None, 'init': {'port': '/dev/ttyTMCM*', 'address': 8, 'axes': ['stig'], 'ustepsize': [2.72e-05], 'rng': [[-14, 7]], 'unit': ['rad'], 'refproc': 'Standard', 'refswitch': {'stig': 0}}, 'parents': ['AntiBacklash Actuators']}, 'Stigmator': {'class': 'actuator.RotationActuator', 'role': 'stigmator', 'affects': ['Camera'], 'children': {'rz': 'AntiBacklash Actuators'}, 'init': {'axis_name': 'stig', 'cycle': 6.283185, 'ref_start': None}, 'metadata': {'POS_COR': -0.698768, 'CALIB': {0.0262: {'x': {'a': 0.2776726566744459, 'b': 0.005243584563638028, 'c': 522.2281261564915, 'd': 483.0808116203732, 'w0': 9.524395314959095}, 'y': {'a': -0.3617690717345442, 'b': -0.03566609269230697, 'c': 1424.134120732775, 'd': 457.3606790990147, 'w0': 9.388879997702146}, 'feature_angle': -3.1416, 'upsample_factor': 5, 'z_least_confusion': 9.836207241448289e-07, 'z_calibration_range': [-9.836207241448289e-07, 8.616379275855172e-06]}}}}, 'AntiBacklash Actuators': {'class': 'actuator.AntiBacklashActuator', 'role': None, 'init': {'backlash': {'stig': 0.02}}, 'children': {'slave': 'Stigmator Actuator'}, 'parents': ['Stigmator']}, 'Optical Focus': {'class': 'smaract.MCS2', 'role': 'focus', 'init': {'locator': 'network:sn:MCS2-00005114', 'param_file': '/usr/share/odemis/meteor-optical-focus.mcs2.tsv', 'ref_on_init': True, 'speed': 0.002, 'accel': 0.002, 'axes': {'z': {'range': [-0.015, 0.015], 'unit': 'm', 'channel': 0}}}, 'metadata': {'FAV_POS_DEACTIVE': {'z': -0.011}, 'FAV_POS_ACTIVE': {'z': -0.005}}, 'affects': ['Camera']}}
	swVersion (RO Attribute)	value: Unknown (Odemis 3.6.0-246-g9ac466a5b-dirty)
	alive (Vigilant Attribute)	 value: {'AntiBacklash Actuators', 'AntiBacklash for Filter Wheel', 'Filter Wheel', 'Light Source', 'Optical Actuators', 'Optical Focus', 'Optical Objective', 'Stigmator', 'Stigmator Actuator'}
	dependencies (Vigilant Attribute)	 value: {}
	ghosts (Vigilant Attribute)	 value: {'Camera': HwError('Failed to find Andor camera 0, check that it is turned on and connected to the computer.'), 'Electron-Beam': 'unloaded', 'Electron-Detector': 'unloaded', 'Electron-Focus': 'unloaded', 'FIBSEM': HwError("Failed to connect to autoscript server 'localhost'. Check that the uri is correct and autoscript server is connected to the network. cannot connect to ('localhost', 4242): [Errno 111] Connection refused PYRO 5: 5.12,\nmsgpack-numpy: 0.4.7.1,\nmsgpack_numpy_file: /usr/lib/python3/dist-packages/msgpack_numpy.py,\nmsgpack version: 0.6.2\nmsgpack_file: /usr/lib/python3/dist-packages/msgpack/__init__.py\n\nThis is likely a dependencies issue:\nmake sure you have the following matching versions of msgpack and msgpack-numpy installed:\nmsgpack==0.5.6, msgpack-numpy==0.4.4\nor\nmsgpack==1.0.3 msgpack-numpy==0.4.8\n"), 'Ion-Beam': 'unloaded', 'Ion-Detector': 'unloaded', 'Ion-Focus': 'unloaded', 'Stage': 'unloaded'}
	state (RO Vigilant Attribute)	 value: running

Component 'Stigmator Actuator':
	role: None
	affects:
	axes (RO Attribute)
		stig:	Axis in -14.0000032 -> 7.0000016 rad
	hwVersion (RO Attribute)	value: TMCM-6110 (firmware 1.42)
	swVersion (RO Attribute)	value: 3.6.0-246-g9ac466a5b-dirty (serial driver: Unknown)
	dependencies (Vigilant Attribute)	 value: {}
	position (RO Vigilant Attribute)	 value: {'stig': -0.6987680000000001} (unit: m)	{'stig': -40.03645725879751°}
	referenced (RO Vigilant Attribute)	 value: {'stig': True}
	speed (RO Vigilant Attribute)	 value: {'stig': 13.28125} (unit: m/s)
	state (RO Vigilant Attribute)	 value: running
