# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1523, 741)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ConnectButton = QtWidgets.QPushButton(self.centralwidget)
        self.ConnectButton.setGeometry(QtCore.QRect(20, 20, 141, 23))
        self.ConnectButton.setObjectName("ConnectButton")
        self.DisconnectButton = QtWidgets.QPushButton(self.centralwidget)
        self.DisconnectButton.setGeometry(QtCore.QRect(180, 20, 151, 23))
        self.DisconnectButton.setObjectName("DisconnectButton")
        self.RefImage = QtWidgets.QPushButton(self.centralwidget)
        self.RefImage.setGeometry(QtCore.QRect(350, 20, 141, 23))
        self.RefImage.setObjectName("RefImage")
        self.ResetImage = QtWidgets.QPushButton(self.centralwidget)
        self.ResetImage.setGeometry(QtCore.QRect(520, 20, 81, 23))
        self.ResetImage.setObjectName("ResetImage")
        self.EB_Image = QtWidgets.QLabel(self.centralwidget)
        self.EB_Image.setGeometry(QtCore.QRect(20, 70, 400, 400))
        self.EB_Image.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.EB_Image.setText("")
        self.EB_Image.setObjectName("EB_Image")
        self.IB_Image = QtWidgets.QLabel(self.centralwidget)
        self.IB_Image.setGeometry(QtCore.QRect(450, 70, 400, 400))
        self.IB_Image.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.IB_Image.setText("")
        self.IB_Image.setObjectName("IB_Image")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 480, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(610, 480, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.CLog = QtWidgets.QLabel(self.centralwidget)
        self.CLog.setGeometry(QtCore.QRect(10, 600, 500, 20))
        self.CLog.setText("")
        self.CLog.setObjectName("CLog")
        self.CLog2 = QtWidgets.QLabel(self.centralwidget)
        self.CLog2.setGeometry(QtCore.QRect(10, 630, 500, 20))
        self.CLog2.setText("")
        self.CLog2.setObjectName("CLog2")
        self.CLog3 = QtWidgets.QLabel(self.centralwidget)
        self.CLog3.setGeometry(QtCore.QRect(10, 660, 500, 20))
        self.CLog3.setText("")
        self.CLog3.setObjectName("CLog3")
        self.CLog4 = QtWidgets.QLabel(self.centralwidget)
        self.CLog4.setGeometry(QtCore.QRect(10, 690, 500, 20))
        self.CLog4.setText("")
        self.CLog4.setObjectName("CLog4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 550, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(890, 70, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(900, 130, 47, 13))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(900, 160, 47, 13))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(900, 220, 71, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(900, 190, 71, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(890, 260, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(900, 290, 51, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(900, 320, 51, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(900, 350, 81, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(900, 380, 81, 16))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(900, 455, 81, 16))
        self.label_15.setObjectName("label_15")
        self.EB_Click = QtWidgets.QPushButton(self.centralwidget)
        self.EB_Click.setGeometry(QtCore.QRect(310, 480, 75, 23))
        self.EB_Click.setObjectName("EB_Click")
        self.IB_click = QtWidgets.QPushButton(self.centralwidget)
        self.IB_click.setGeometry(QtCore.QRect(730, 480, 75, 23))
        self.IB_click.setObjectName("IB_click")
        self.gamma_enabled = QtWidgets.QCheckBox(self.centralwidget)
        self.gamma_enabled.setGeometry(QtCore.QRect(900, 100, 70, 17))
        self.gamma_enabled.setObjectName("gamma_enabled")
        self.gamma_min = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.gamma_min.setGeometry(QtCore.QRect(1030, 125, 62, 22))
        self.gamma_min.setObjectName("gamma_min")
        self.gamma_max = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.gamma_max.setGeometry(QtCore.QRect(1030, 155, 62, 22))
        self.gamma_max.setObjectName("gamma_max")
        self.gamma_scalefactor = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.gamma_scalefactor.setGeometry(QtCore.QRect(1030, 215, 62, 22))
        self.gamma_scalefactor.setObjectName("gamma_scalefactor")
        self.res_width = QtWidgets.QSpinBox(self.centralwidget)
        self.res_width.setGeometry(QtCore.QRect(1030, 290, 60, 22))
        self.res_width.setMaximum(10000)
        self.res_width.setObjectName("res_width")
        self.res_height = QtWidgets.QSpinBox(self.centralwidget)
        self.res_height.setGeometry(QtCore.QRect(1115, 290, 60, 22))
        self.res_height.setMaximum(10000)
        self.res_height.setObjectName("res_height")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1100, 295, 15, 13))
        self.label_5.setObjectName("label_5")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(1095, 325, 81, 16))
        self.label_16.setObjectName("label_16")
        self.autocontrast_enable = QtWidgets.QCheckBox(self.centralwidget)
        self.autocontrast_enable.setGeometry(QtCore.QRect(1030, 350, 70, 17))
        self.autocontrast_enable.setObjectName("autocontrast_enable")
        self.reset_image_settings = QtWidgets.QPushButton(self.centralwidget)
        self.reset_image_settings.setGeometry(QtCore.QRect(920, 525, 161, 23))
        self.reset_image_settings.setObjectName("reset_image_settings")
        self.gamma_threshold = QtWidgets.QSpinBox(self.centralwidget)
        self.gamma_threshold.setGeometry(QtCore.QRect(1030, 185, 61, 22))
        self.gamma_threshold.setMaximum(255)
        self.gamma_threshold.setObjectName("gamma_threshold")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(900, 485, 111, 16))
        self.label_17.setObjectName("label_17")
        self.hfw_box = QtWidgets.QSpinBox(self.centralwidget)
        self.hfw_box.setGeometry(QtCore.QRect(1030, 485, 61, 22))
        self.hfw_box.setMaximum(5000)
        self.hfw_box.setObjectName("hfw_box")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(1110, 490, 47, 13))
        self.label_18.setObjectName("label_18")
        self.dwell_time_setting = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dwell_time_setting.setGeometry(QtCore.QRect(1030, 320, 62, 22))
        self.dwell_time_setting.setObjectName("dwell_time_setting")
        self.open_filepath = QtWidgets.QToolButton(self.centralwidget)
        self.open_filepath.setGeometry(QtCore.QRect(1030, 380, 25, 19))
        self.open_filepath.setObjectName("open_filepath")
        self.savepath_text = QtWidgets.QLabel(self.centralwidget)
        self.savepath_text.setGeometry(QtCore.QRect(1070, 380, 241, 16))
        self.savepath_text.setText("")
        self.savepath_text.setObjectName("savepath_text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1523, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ConnectButton.setText(_translate("MainWindow", "Connect to Microscope"))
        self.DisconnectButton.setText(_translate("MainWindow", "Disconnect from Microscope"))
        self.RefImage.setText(_translate("MainWindow", "Take Reference Images"))
        self.ResetImage.setText(_translate("MainWindow", "Reset Images"))
        self.label.setText(_translate("MainWindow", "Electron Beam"))
        self.label_2.setText(_translate("MainWindow", "Ion Beam"))
        self.label_3.setText(_translate("MainWindow", "Console Log"))
        self.label_4.setText(_translate("MainWindow", "Gamma Settings"))
        self.label_6.setText(_translate("MainWindow", "Min"))
        self.label_7.setText(_translate("MainWindow", "Max"))
        self.label_8.setText(_translate("MainWindow", "Scale Factor"))
        self.label_9.setText(_translate("MainWindow", "Threshold"))
        self.label_10.setText(_translate("MainWindow", "Image Settings"))
        self.label_11.setText(_translate("MainWindow", "Resolution"))
        self.label_12.setText(_translate("MainWindow", "Dwell time"))
        self.label_13.setText(_translate("MainWindow", "Autocontrast"))
        self.label_14.setText(_translate("MainWindow", "Save Path"))
        self.label_15.setText(_translate("MainWindow", "Reduced Area"))
        self.EB_Click.setText(_translate("MainWindow", "Take Image"))
        self.IB_click.setText(_translate("MainWindow", "Take Image"))
        self.gamma_enabled.setText(_translate("MainWindow", "Enabled"))
        self.label_5.setText(_translate("MainWindow", "x"))
        self.label_16.setText(_translate("MainWindow", "microseconds"))
        self.autocontrast_enable.setText(_translate("MainWindow", "Enabled"))
        self.reset_image_settings.setText(_translate("MainWindow", "Reset To Default Settings"))
        self.label_17.setText(_translate("MainWindow", "Horizontal Field Width"))
        self.label_18.setText(_translate("MainWindow", "microns"))
        self.open_filepath.setText(_translate("MainWindow", "..."))
