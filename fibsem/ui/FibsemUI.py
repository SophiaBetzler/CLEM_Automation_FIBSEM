import napari
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

import fibsem
from fibsem.microscope import (
    DemoMicroscope,
    FibsemMicroscope,
    TescanMicroscope,
    ThermoMicroscope,
)
from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.ui.FibsemAlignmentWidget import FibsemAlignmentWidget
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemImageViewer import FibsemImageViewer
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.qtdesigner_files import FibsemUI

# TODO: add calibrate manipulator procedure for thermo here

class FibsemUI(FibsemUI.Ui_MainWindow, QtWidgets.QMainWindow):

    def __init__(self, viewer: napari.Viewer):
        super(FibsemUI, self).__init__()
        self.setupUi(self)

        self.label_title.setText(f"OpenFIBSEM v{fibsem.__version__}")

        self.viewer = viewer
        # self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        # self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.microscope: FibsemMicroscope = None
        self.settings: MicroscopeSettings = None

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: FibsemMillingWidget = None
        self.alignment_widget: FibsemAlignmentWidget = None
        self.manipulator_widget: FibsemManipulatorWidget = None
        self.image_viewer: FibsemImageViewer = None

        self.minimap_widget: FibsemMinimapWidget = None

        self.system_widget = FibsemSystemSetupWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                parent=self,
            )
        
        self.tabWidget.addTab(self.system_widget, "Connection")
        self.setup_connections()
        self.ref_current = None
        self.update_ui()


    def setup_connections(self):
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)
        if self.manipulator_widget is not None:
            self.actionManipulator_Positions_Calibration.triggered.connect(self.manipulator_widget.calibrate_manipulator_positions)
        self.actionOpen_Minimap.triggered.connect(self.open_minimap_widget)

    def open_minimap_widget(self):
        if self.microscope is None:
            napari.utils.notifications.show_warning("Please connect to a microscope first... [No Microscope Connected]")
            return

        if self.movement_widget is None:
            napari.utils.notifications.show_warning("Please connect to a microscope first... [No Movement Widget]")
            return

        self.viewer_minimap = napari.Viewer(ndisplay=2)
        self.minimap_widget = FibsemMinimapWidget(viewer=self.viewer_minimap, parent=self)
        self.viewer_minimap.window.add_dock_widget(
            widget=self.minimap_widget, 
            area="right", 
            add_vertical_stretch=True, 
            name="OpenFIBSEM Minimap"
        )
        napari.run(max_loop_level=2)

    def align_currents(self):
        second_viewer = napari.Viewer()
        self.alignment_widget = FibsemAlignmentWidget(settings=self.settings, microscope=self.microscope, viewer=second_viewer, parent = self)
        second_viewer.window.add_dock_widget(self.alignment_widget, name='Beam Alignment', area='right')
        self.alignment_widget.destroyed.connect(self.reset_currents)

    def reset_currents(self):
        if isinstance(self.microscope, ThermoMicroscope)or isinstance(self.microscope, DemoMicroscope):
            self.microscope.set("current", float(self.ref_current), BeamType.ION)
        if isinstance(self.microscope, TescanMicroscope):
            self.microscope.set("preset", self.ref_current, BeamType.ION)


    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)
        self.tabWidget.setTabVisible(4, _microscope_connected)
        self.actionOpen_Minimap.setVisible(_microscope_connected)
        self.actionManipulator_Positions_Calibration.setVisible(_microscope_connected)


    def connect_to_microscope(self):
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):
        
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()
        self.image_widget = None
        self.movement_widget = None
        self.milling_widget = None

    def update_microscope_ui(self):

        if self.microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,
                viewer=self.viewer,
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                viewer=self.viewer,
                parent=self,
            )
            self.milling_widget = FibsemMillingWidget(
                microscope=self.microscope,
                viewer=self.viewer,
                parent=self,
            )
            if self.microscope.system.manipulator.enabled:
                self.manipulator_widget = FibsemManipulatorWidget(
                    microscope=self.microscope,
                    settings=self.settings,
                    viewer=self.viewer,
                    image_widget=self.image_widget,
                    parent=self,
                )
            else:
                self.manipulator_widget = None
  


            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.tabWidget.addTab(self.milling_widget, "Milling")

            if self.microscope.system.manipulator.enabled:
                self.tabWidget.addTab(self.manipulator_widget, "Manipulator")

            self.system_widget.image_widget = self.image_widget
            self.system_widget.milling_widget = self.milling_widget

        else:
            if self.image_widget is None:
                return
            
            # remove tabs
            self.tabWidget.removeTab(4)
            self.tabWidget.removeTab(3)
            self.tabWidget.removeTab(2)
            self.tabWidget.removeTab(1)
            self.image_widget.clear_viewer()
            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            if self.manipulator_widget is not None:
                self.manipulator_widget.deleteLater() 



def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, 
                                  area="right", 
                                  add_vertical_stretch=True, 
                                  name=f"OpenFIBSEM v{fibsem.__version__}")
    napari.run()


if __name__ == "__main__":
    main()



