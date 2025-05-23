import logging
import os
import yaml
from pathlib import Path
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemPositionsWidget, FibsemMovementWidget
from fibsem.ui.utils import open_save_file_dialog, open_existing_file_dialog


class FibsemPositionsWidget(FibsemPositionsWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        movement_widget: FibsemMovementWidget = None,
        image_widget: FibsemImageSettingsWidget = None,
        parent=None,
    ):
        super(FibsemPositionsWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.movement_widget = movement_widget
        self.image_widget = image_widget
        self.setup_connections()

        self.positions = []

    def setup_connections(self):
        self.comboBox_positions.currentIndexChanged.connect(self.select_position)
        self.pushButton_save_position.clicked.connect(self.add_position)
        self.pushButton_remove_position.clicked.connect(self.delete_position)
        self.pushButton_go_to.clicked.connect(self.go_to_position)
        self.pushButton_export.clicked.connect(self.export_positions)
        self.pushButton_import.clicked.connect(self.import_positions)

    def select_position(self):
        if self.comboBox_positions.currentIndex() != -1:
            position = self.positions[self.comboBox_positions.currentIndex()]
            self.label_current_position.setText(f"x={position.x*constants.METRE_TO_MILLIMETRE:.3f}, y={position.y*constants.METRE_TO_MILLIMETRE:.3f}, z={position.z*constants.METRE_TO_MILLIMETRE:.3f}, r={position.r*constants.RADIANS_TO_DEGREES:.1f}, t={position.t*constants.RADIANS_TO_DEGREES:.1f}")

    def add_position(self):
        position = self.microscope.get_stage_position()
        name = self.lineEdit_position_name.text()
        if name == "":
            napari.utils.notifications.show_warning("Please enter a name for the position")
            return
        position.name = name
        self.positions.append(position)
        self.comboBox_positions.addItem(name)
        self.comboBox_positions.setCurrentIndex(self.comboBox_positions.count() - 1)
        self.lineEdit_position_name.setText("")
        logging.info(f"Added position {name}")

    def delete_position(self):
        del self.positions[self.comboBox_positions.currentIndex()]
        name = self.comboBox_positions.currentIndex()
        self.comboBox_positions.removeItem(self.comboBox_positions.currentIndex())
        logging.info(f"Removed position {name}")

    def go_to_position(self):
        self.microscope.move_stage_absolute(self.positions[self.comboBox_positions.currentIndex()])
        self.movement_widget.update_ui()
        self.image_widget.acquire_reference_images()
        logging.info(f"Moved to position {self.comboBox_positions.currentIndex()}")

    def export_positions(self):
        protocol_path = open_save_file_dialog(msg="Select or create file")
        if protocol_path == '':
            return
        dict_position = []
        for position in self.positions:
            dict_position.append(position.to_dict())
        with open(os.path.join(Path(protocol_path).with_suffix(".yaml")), "w") as f:
            yaml.safe_dump(dict_position, f, indent=4, default_flow_style=False)

        logging.info("Positions saved to file")


    def import_positions(self):
        protocol_path = open_existing_file_dialog(msg="Select or create file")
        if protocol_path == '':
            napari.utils.notifications.show_info("No file selected, positions not loaded")
            return
        with open(protocol_path, "r") as f:
            dict_positions = yaml.safe_load(f)
        for dict_position in dict_positions:
            position = FibsemStagePosition.from_dict(dict_position)
            self.positions.append(position)
            self.comboBox_positions.addItem(position.name)


def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemPositionsWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
