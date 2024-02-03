import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap

import ui as ui
from classify import classify

path=None
class LoadUI(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.openButton.clicked.connect(self.loadImage)
        self.classifyButton.clicked.connect(self.classifyImage)

    def loadImage(self):
        self.imageLabel.setText("Loading...")
        imagePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open','.','*.png *.jpg *.bmp')
        imagePath = imagePath[0]
        self.path = imagePath
        pixmapImage = QPixmap(imagePath)
        # pixmapImage = pixmapImage.scaled(790,500)
        self.imageLabel.resize(pixmapImage.width(), pixmapImage.height())
        self.imageLabel.setPixmap(pixmapImage)
        self.outputLabel.setText(imagePath)
        self.classifyButton.setEnabled(True)
    
    def classifyImage(self):
        self.outputLabel.setText("Classifying...")
        # print(self.path)
        classify(self.path)
        currentDirectory = os.getcwd()
        resultTextPath = os.path.join(currentDirectory, "result", "result_out.txt")
        resultImagePath = os.path.join(currentDirectory, "result", "result.jpg")
        # pixmapImage = QPixmap('G:\\tsd\\gui\\result\\result.jpg')
        # # pixmapImage = pixmapImage.scaled(790,500)
        # self.imageLabel.resize(pixmapImage.width(), pixmapImage.height())
        # self.imageLabel.setPixmap(pixmapImage)
        # with open("result\\result_out.txt","r") as f:
        #     op = [line.strip() for line in f.readlines()]
        #     print(op)
        pixmapImage = None
        text = open(resultTextPath).read()
        if(len(text)==0):
            self.outputLabel.setText("No Trafic Signs Detected.")
            print("No Trafic Signs Detected.")
        else:
            if(text.find("Error")==0):
                pixmapImage = QPixmap(self.path)
                self.imageLabel.resize(pixmapImage.width(), pixmapImage.height())
                self.imageLabel.setPixmap(pixmapImage)
                self.outputLabel.setText(text)
            else:
                pixmapImage = QPixmap(resultImagePath)
                self.imageLabel.resize(pixmapImage.width(), pixmapImage.height())
                self.imageLabel.setPixmap(pixmapImage)
                self.outputLabel.setText(text)




def main():
    app = QtWidgets.QApplication(sys.argv)
    window = LoadUI()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()