from Basic_Functions import BasicFunctions
from fibsem import utils, structures, microscope
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button
#from autoscript_sdb_microscope_client import SdbMicroscopeClient
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox
)
from PyQt5.QtCore import Qt


class AutomatedTriCoincidence():
    def __init__(self):
        self.tricoincidence = TriCoincidence()
        self.app = QApplication(sys.argv)
        self.window = GUIforTriCoincidence(self)
        self.run()

    def run(self):
        self.window.show()
        self.app.exec()


class TriCoincidence(BasicFunctions):
    def __init__(self):
        super().__init__()
        if self.tool != 'Arctis':
            raise RuntimeError("This is not the right tool to run the automated tricoincidence routine.")
        self.grid_numbers, self.available_grids  = self.autoloader_control()

    def run_automated_process(self):
        result = self.run_add_position()
        print("Result from add position:", result)


    def run_add_position(self):
        """
        This function is called once the user adds an item to the table. It will record the current stage position,
        objective focus, emission settings, open an image to draw the ROI for the FL targeting.
        """

        for i in len(self.available_grids):
            if self.available_grids[i].state == 'Loaded':
                current_grid_number = i

        current_stage_position = microscope.get_stage_position()

        self.thermo_microscope.imaging.set_active_view(3)
        self.thermo_microscope.imaging.set_active_device(ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE)
        current_objective_focus = self.thermo_microscope.detector.camera_settings.focus.value

        fl_settings = {'emission_color': self.thermo_microscope.detector.camera_settings.emission.type.value,
                       'filter_setting': self.thermo_microscope.detector.camera_settings.filter.value,
                       'exposure_time': self.thermo_microscope.detector.camera_settings.exposure_time.value,
                       'binning': self.thermo_microscope.detector.camera_settings.binning.value,
        }

        fl_roi = self.define_roi()

        return current_grid_number, current_stage_position, current_objective_focus, fl_settings, fl_roi

    def define_roi(self):
        """
        Define a ROI in the fluorescence image which will be used to calculate the average.
        This script will take the currently displayed image in the 3 view of XT as reference.
        """
        roi_coords = [None]  # Use a mutable object to capture updates

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            roi_coords[0] = (xmin, xmax, ymin, ymax)

        def on_ok_clicked(event):
            if roi_coords[0] is not None:
                print(
                    f"Final ROI confirmed: x={roi_coords[0][0]}:{roi_coords[0][1]}, y={roi_coords[0][2]}:{roi_coords[0][3]}")
                plt.close(fig)
            else:
                print("No ROI selected yet.")

        # self.microscope.imaging.set_active_view(3)
        # image = self.microscope.imaging.get_image()
        image = np.random.rand(1024, 1024)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image.data, cmap='gray')
        ax.set_title("Draw ROI, then click OK to confirm")

        selector = RectangleSelector(
            ax, onselect,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True)

        ok_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        ok_button = Button(ok_ax, 'OK')
        ok_button.on_clicked(on_ok_clicked)
        plt.show()

        if roi_coords:
            return roi_coords[0]
        else:
            print("No ROI was selected.")
            return None

class GUIforTriCoincidence(QWidget):
    def __init__(self, AutomatedTriCoincidence):
        super().__init__()
        self.automated_tricoincidence = AutomatedTriCoincidence
        self.setWindowTitle("Setup of the TriCoincidence Routine")

        # === POSITION SETUP SECTION ===
        position_setup_title = QLabel("Setup of the Positions")
        position_setup_title.setStyleSheet("font-weight: bold; font-size: 14px")

        position_setup_description = QLabel("Please select all position of interest on all grids. Make sure to adjust"
                                            "the optical focus and click 'Add'. \nPositions can be edited or deleted"
                                            " by selection the respective row.")
        position_setup_description.setStyleSheet("color: gray; font-size: 11px")


        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Grid", "Stage Position", "Objective Focus", "ROI", "Status"])

        # Vertical side buttons (now: Add, Edit, Delete)
        side_buttons = QVBoxLayout()

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_position)
        side_buttons.addWidget(add_button)

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_selected)
        side_buttons.addWidget(edit_button)

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_selected)
        side_buttons.addWidget(delete_button)

        import_button = QPushButton("Import")
        import_button.clicked.connect(self.delete_selected)
        side_buttons.addWidget(import_button)

        display_button = QPushButton("Display")
        display_button.clicked.connect(self.delete_selected)
        side_buttons.addWidget(display_button)


        side_buttons.addStretch()  # Push buttons to the top

        # Table layout only (no bottom row buttons anymore)
        table_layout = QVBoxLayout()
        table_layout.addWidget(self.table)

        # Combine table + side buttons
        top_row_layout = QHBoxLayout()
        top_row_layout.addLayout(table_layout, stretch=4)
        top_row_layout.addLayout(side_buttons, stretch=1)

        # === DIVIDER ===
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        # === BOTTOM SECTION ===
        bottom_title = QLabel("Configuration")
        bottom_title.setStyleSheet("font-weight: bold; font-size: 14px")

        input_layout = QVBoxLayout()
        input_layout.setAlignment(Qt.AlignLeft)  # Align left

        input1_label = QLabel("Input 1:")
        input1_label.setFixedWidth(60)
        self.input1 = QSpinBox()
        self.input1.setValue(0)
        self.input1.setMaximumWidth(100)

        input2_label = QLabel("Input 2:")
        input2_label.setFixedWidth(60)
        self.input2 = QSpinBox()
        self.input2.setValue(0)
        self.input2.setMaximumWidth(100)

        # Stack vertically
        input_layout.addWidget(input1_label)
        input_layout.addWidget(self.input1)
        input_layout.addSpacing(10)
        input_layout.addWidget(input2_label)
        input_layout.addWidget(self.input2)

        # Bottom action buttons
        bottom_buttons = QHBoxLayout()
        for label in ["Start", "Abort", "Export"]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, l=label: self.bottom_button_clicked(l))
            bottom_buttons.addWidget(btn)

        # Bottom layout
        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(bottom_title)
        #bottom_layout.addWidget(bottom_description)
        bottom_layout.addLayout(input_layout)
        bottom_layout.addLayout(bottom_buttons)

        # === FINAL LAYOUT ===
        main_layout = QVBoxLayout()
        main_layout.addWidget(position_setup_title)
        main_layout.addWidget(position_setup_description)
        main_layout.addLayout(top_row_layout)
        main_layout.addWidget(divider)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def add_position(self):
        row = self.table.rowCount()
        result = self.automated_tricoincidence.tricoincidence.run_add_position()
        print(result)
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("New Name"))
        self.table.setItem(row, 1, QTableWidgetItem(str(self.input1.value())))
        self.table.setItem(row, 2, QTableWidgetItem("✓"))

    def delete_selected(self):
        selected = self.table.selectionModel().selectedRows()
        for index in sorted(selected, key=lambda x: x.row(), reverse=True):
            self.table.removeRow(index.row())

    def edit_selected(self):
        selected = self.table.selectionModel().selectedRows()
        if selected:
            row = selected[0].row()
            self.table.setItem(row, 0, QTableWidgetItem("Edited Name"))
            self.table.setItem(row, 1, QTableWidgetItem(str(self.input2.value())))
            self.table.setItem(row, 2, QTableWidgetItem("✓"))

    def side_button_clicked(self, label):
        print(f"Side button {label} clicked")

    def bottom_button_clicked(self, label):
        print(f"{label} clicked — Input1: {self.input1.value()}, Input2: {self.input2.value()}")

