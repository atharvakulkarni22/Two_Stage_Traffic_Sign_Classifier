from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 700)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setMaximumSize(QtCore.QSize(1280, 700))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        MainWindow.setFont(font)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(-1, -1,1281, 701))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setSpacing(10)
        self.gridLayout.setObjectName("gridLayout")

        self.classifyButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.classifyButton.setFont(font)
        self.classifyButton.setObjectName("classifyButton")
        self.classifyButton.setEnabled(False)
        self.gridLayout.addWidget(self.classifyButton, 2, 0, 1, 1)

        self.openButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.openButton.setFont(font)
        self.openButton.setObjectName("openButton")
        self.gridLayout.addWidget(self.openButton, 1, 0, 1, 1)

        self.imageLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.imageLabel.setStyleSheet("border: 1px solid gray; background-color: rgb(255, 255, 255);")
        self.imageLabel.setText("")
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setMaximumSize(1250, 650)
        self.gridLayout.addWidget(self.imageLabel, 0, 0, 1, 2)

        self.outputLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        font.setPointSize(10)
        self.outputLabel.setFont(font)
        self.outputLabel.setStyleSheet("border: 1px solid black; background-color: rgb(255, 255, 255);")
        self.outputLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.outputLabel.setObjectName("outputLabel")
        self.gridLayout.addWidget(self.outputLabel, 1, 1, 2, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Classifier"))
        self.classifyButton.setText(_translate("MainWindow", "Classify Image"))
        self.openButton.setText(_translate("MainWindow", "Open Image"))
        self.outputLabel.setText(_translate("MainWindow", "Output will be show here."))
