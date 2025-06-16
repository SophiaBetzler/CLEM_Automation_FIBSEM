from Basic_Functions import OverArch
from fibsem import utils, structures, microscope
from fibsem.structures import FibsemStagePosition
from GIS_Sputter_Setup import GisSputterAutomation
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import copy
import queue
import platform
import time
pc_type = platform.system()
if pc_type == 'Windows':
    matplotlib.use('Qt5Agg')
elif pc_type == 'Darwin':
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

#from autoscript_sdb_microscope_client import SdbMicroscopeClient
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QSpinBox, QCheckBox, QGridLayout, QFileDialog,
    QDialog
)
from PyQt5.QtCore import Qt, QEventLoop, QObject, pyqtSignal, QThread, QTimer, QMetaObject
from PyQt5.QtGui import QBrush, QColor, QIcon
from PyQt5.QtCore import Qt
from collections import namedtuple
import json
import matplotlib.image as mpimg
import os
from Imaging import Imaging
import tifffile
import cv2
import threading
from datetime import datetime
import statistics
from collections import deque
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QGridLayout, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem
)
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
from PyQt5.QtCore import QTimer, QRectF, Qt, QPointF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class AutomatedTriCoincidenceGUI:
    """
    Sole purpose of this class is to open the GUI controlling the process.
    """
    def __init__(self, oa, coin, mode):
        self.oa = oa
        self.mode = mode
        self.coincidence = coin
        self.app = QApplication(sys.argv)
        self.auto_gui_window = AutoCoincidenceGUI(self.oa)
        self.manual_gui_window = ManualCoincidenceGUI()
        self.run()

    def run(self):
        if self.mode == 'manual':
            self.manual_gui_window.show()
        elif self.mode == 'auto':
            self.auto_gui_window.show()
        else:
            self.error_messagebox("No valid mode selected. Options are 'manual' or 'auto'.")
        self.app.exec()


    def error_messagebox(self, text):
        box = QMessageBox()
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Warning")
        box.setText(text)
        box.setStandardButtons(QMessageBox.Abort)
        choice = box.exec_()
        if choice == QMessageBox.Abort:
            return



HANDLE_SIZE = 6

class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect):
        super().__init__(rect)
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setBrush(QColor(255, 0, 0, 50))
        self.setPen(QPen(Qt.red, 2))
        self.handles = {}
        self.handle_size = HANDLE_SIZE
        self.handle_selected = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None

    def boundingRect(self):
        o = self.handle_size / 2
        return self.rect().adjusted(-o, -o, o, o)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        self.updateHandlesPos()
        painter.setBrush(QColor(0, 255, 0))
        for handle, rect in self.handles.items():
            painter.drawRect(rect)

    def updateHandlesPos(self):
        s = self.handle_size
        b = self.rect()
        self.handles['tl'] = QRectF(b.left()-s/2, b.top()-s/2, s, s)
        self.handles['tr'] = QRectF(b.right()-s/2, b.top()-s/2, s, s)
        self.handles['bl'] = QRectF(b.left()-s/2, b.bottom()-s/2, s, s)
        self.handles['br'] = QRectF(b.right()-s/2, b.bottom()-s/2, s, s)

    def mousePressEvent(self, event):
        for k, r in self.handles.items():
            if r.contains(event.pos()):
                self.handle_selected = k
                break
        self.mouse_press_pos = event.pos()
        self.mouse_press_rect = self.rect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.handle_selected is not None:
            diff = event.pos() - self.mouse_press_pos
            r = QRectF(self.mouse_press_rect)
            if self.handle_selected == 'tl':
                r.setTopLeft(r.topLeft() + diff)
            elif self.handle_selected == 'tr':
                r.setTopRight(r.topRight() + diff)
            elif self.handle_selected == 'bl':
                r.setBottomLeft(r.bottomLeft() + diff)
            elif self.handle_selected == 'br':
                r.setBottomRight(r.bottomRight() + diff)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.handle_selected = None
        super().mouseReleaseEvent(event)

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, with_roi=False):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.roi = None
        if with_roi:
            self.roi = ResizableRectItem(QRectF(100, 100, 120, 80))
            self.scene.addItem(self.roi)
        self.image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.vmin, self.vmax = 0, 255
        self.setFixedSize(520, 520)
        self.scale_factor = 1.15

    def update_image(self, new_image):
        self.image = new_image
        img = np.clip((new_image - self.vmin) / (self.vmax - self.vmin) * 255, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_item.setPixmap(pixmap)

    def get_roi_rect(self):
        if self.roi is None:
            return None
        rect = self.roi.rect()
        topLeft = self.roi.scenePos()
        return QRectF(topLeft.x(), topLeft.y(), rect.width(), rect.height())

    def get_roi_data(self):
        roi = self.get_roi_rect()
        if roi is None:
            return np.array([])
        x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
        return self.image[y:y+h, x:x+w] if w > 0 and h > 0 else np.array([])

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)

class ManualCoincidenceGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.static_view = ZoomableGraphicsView(with_roi=True)
        self.dynamic_view = ZoomableGraphicsView(with_roi=False)

        self.figure, (self.ax_line, self.ax_hist) = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)

        self.min_input = QLineEdit("0")
        self.max_input = QLineEdit("255")
        self.update_button = QPushButton("Update Display Range")
        self.update_button.clicked.connect(self.update_display_range)

        self.capture_button = QPushButton("Capture Static Image")
        self.capture_button.clicked.connect(self.capture_static_image)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.stop_button = QPushButton("Stop")

        self.start_button.clicked.connect(self.start_timer)
        self.pause_button.clicked.connect(self.pause_timer)
        self.resume_button.clicked.connect(self.resume_timer)
        self.stop_button.clicked.connect(self.stop_timer)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.static_view)
        image_layout.addWidget(self.dynamic_view)

        control_layout = QGridLayout()
        control_layout.addWidget(QLabel("Min:"), 0, 0)
        control_layout.addWidget(self.min_input, 0, 1)
        control_layout.addWidget(QLabel("Max:"), 0, 2)
        control_layout.addWidget(self.max_input, 0, 3)
        control_layout.addWidget(self.update_button, 0, 4)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.resume_button)
        button_layout.addWidget(self.stop_button)

        vbox = QVBoxLayout()
        vbox.addLayout(image_layout)
        vbox.addWidget(self.canvas)
        vbox.addLayout(control_layout)
        vbox.addLayout(button_layout)
        self.setLayout(vbox)

        self.vmin, self.vmax = 0, 255
        self.static_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.dynamic_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.update_static_image()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dynamic)

    def update_static_image(self):
        self.static_view.update_image(self.static_img)

    def update_dynamic(self):
        self.dynamic_img = np.roll(self.dynamic_img, -1, axis=1)
        self.dynamic_img[:, -1] = np.random.randint(0, 255, size=(512,))
        self.dynamic_view.update_image(self.dynamic_img)
        self.update_plot()

    def update_plot(self):
        roi = self.static_view.get_roi_rect()
        if roi is None:
            return
        x, y, w, h = int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height())
        roi_data = self.dynamic_view.image[y:y+h, x:x+w] if w > 0 and h > 0 else np.array([])

        if roi_data.size == 0:
            return
        line_profile = roi_data.mean(axis=0)

        self.ax_line.clear()
        self.ax_line.plot(line_profile)
        self.ax_line.set_title("ROI Line Profile (Dynamic Image)")

        self.ax_hist.clear()
        self.ax_hist.hist(self.dynamic_view.image.flatten(), bins=50, color='gray')
        self.ax_hist.set_title("Dynamic Image Histogram")

        self.canvas.draw()

    def update_display_range(self):
        try:
            self.vmin = float(self.min_input.text())
            self.vmax = float(self.max_input.text())
            self.static_view.vmin = self.vmin
            self.dynamic_view.vmin = self.vmin
            self.static_view.vmax = self.vmax
            self.dynamic_view.vmax = self.vmax
            self.update_static_image()
            self.dynamic_view.update_image(self.dynamic_img)
        except ValueError:
            pass

    def start_timer(self):
        self.timer.start(100)

    def pause_timer(self):
        self.timer.stop()

    def resume_timer(self):
        self.timer.start(100)

    def stop_timer(self):
        self.timer.stop()
        self.dynamic_img = np.zeros_like(self.dynamic_img)
        self.dynamic_view.update_image(self.dynamic_img)



    def capture_static_image(self):
        # Simulate a new static image (replace with real acquisition if needed)
        self.static_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.update_static_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = QMainWindow()
    widget = ImageViewer()
    win.setCentralWidget(widget)
    win.setWindowTitle('ROI on Static, Plot from Dynamic Image')
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())
