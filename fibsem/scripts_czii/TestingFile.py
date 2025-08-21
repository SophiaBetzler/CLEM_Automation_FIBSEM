import sys
from PyQt5 import QtWidgets


class ParameterWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main layout for the window
        main_layout = QtWidgets.QVBoxLayout()

        # Tilts group with min, max, and steps
        tilt_group = QtWidgets.QGroupBox("Tilts")
        tilt_layout = QtWidgets.QFormLayout()
        self.tilt_min_input = QtWidgets.QLineEdit()
        self.tilt_max_input = QtWidgets.QLineEdit()
        self.tilt_steps_input = QtWidgets.QLineEdit()
        tilt_layout.addRow("Minimum:", self.tilt_min_input)
        tilt_layout.addRow("Maximum:", self.tilt_max_input)
        tilt_layout.addRow("Steps:", self.tilt_steps_input)
        tilt_group.setLayout(tilt_layout)

        # Biases group with min, max, and steps
        bias_group = QtWidgets.QGroupBox("Biases")
        bias_layout = QtWidgets.QFormLayout()
        self.bias_min_input = QtWidgets.QLineEdit()
        self.bias_max_input = QtWidgets.QLineEdit()
        self.bias_steps_input = QtWidgets.QLineEdit()
        bias_layout.addRow("Minimum:", self.bias_min_input)
        bias_layout.addRow("Maximum:", self.bias_max_input)
        bias_layout.addRow("Steps:", self.bias_steps_input)
        bias_group.setLayout(bias_layout)

        # Voltages group with min, max, and steps
        voltage_group = QtWidgets.QGroupBox("Voltages")
        voltage_layout = QtWidgets.QFormLayout()
        self.voltage_min_input = QtWidgets.QLineEdit()
        self.voltage_max_input = QtWidgets.QLineEdit()
        self.voltage_steps_input = QtWidgets.QLineEdit()
        voltage_layout.addRow("Minimum:", self.voltage_min_input)
        voltage_layout.addRow("Maximum:", self.voltage_max_input)
        voltage_layout.addRow("Steps:", self.voltage_steps_input)
        voltage_group.setLayout(voltage_layout)

        # Submit button
        self.submit_button = QtWidgets.QPushButton("Submit")
        self.submit_button.clicked.connect(self.onSubmit)

        # Add groups and button to the main layout
        main_layout.addWidget(tilt_group)
        main_layout.addWidget(bias_group)
        main_layout.addWidget(voltage_group)
        main_layout.addWidget(self.submit_button)

        self.setLayout(main_layout)
        self.setWindowTitle("Parameter Setup")
        self.show()

    def onSubmit(self):
        # Retrieve tilts values
        tilt_min = self.tilt_min_input.text()
        tilt_max = self.tilt_max_input.text()
        tilt_steps = self.tilt_steps_input.text()

        # Retrieve biases values
        bias_min = self.bias_min_input.text()
        bias_max = self.bias_max_input.text()
        bias_steps = self.bias_steps_input.text()

        # Retrieve voltages values
        voltage_min = self.voltage_min_input.text()
        voltage_max = self.voltage_max_input.text()
        voltage_steps = self.voltage_steps_input.text()

        print(type(tilt_min))
        # For demonstration, print the values to the console
        print("Tilts: {} to {}, Steps: {}".format(tilt_min, tilt_max, tilt_steps))
        print("Biases: {} to {}, Steps: {}".format(bias_min, bias_max, bias_steps))
        print("Voltages: {} to {}, Steps: {}".format(voltage_min, voltage_max, voltage_steps))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ParameterWindow()
    sys.exit(app.exec_())
