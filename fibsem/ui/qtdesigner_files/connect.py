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
        MainWindow.resize(856, 540)
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
        self.EB_Image.setGeometry(QtCore.QRect(20, 100, 281, 201))
        self.EB_Image.setObjectName("EB_Image")
        self.IB_Image = QtWidgets.QLabel(self.centralwidget)
        self.IB_Image.setGeometry(QtCore.QRect(390, 100, 281, 191))
        self.IB_Image.setObjectName("IB_Image")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 310, 141, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(390, 310, 141, 16))
        self.label_2.setObjectName("label_2")
        self.CLog = QtWidgets.QLabel(self.centralwidget)
        self.CLog.setGeometry(QtCore.QRect(20, 420, 511, 16))
        self.CLog.setObjectName("CLog")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 856, 21))
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
        self.EB_Image.setText(_translate("MainWindow", "TextLabel"))
        self.IB_Image.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "Electron Beam"))
        self.label_2.setText(_translate("MainWindow", "Ion Beam"))
        self.CLog.setText(_translate("MainWindow", "Console Log"))
