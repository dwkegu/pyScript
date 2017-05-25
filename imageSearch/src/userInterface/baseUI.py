import sys
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage
from PyQt5.QtWidgets import QApplication,QWidget, QPushButton, QFileDialog, QLineEdit, QLabel
from PyQt5 import QtCore
import os
import sys
class user(QWidget):
    '''
    main windows to presentation the UI
    '''
    def __init__(self, retrieval, addImage):
        super().__init__()
        self._count = 0
        self.setWindowTitle("基于深度学习的以图搜图系统")
        self.resize(1600,1000)
        self.move(400, 300)
        path = sys.path[0]
        self.setWindowIcon(QIcon(os.path.join(path, "../sources/icon/image-search.png")))
        self.imageLine = QLineEdit(self)
        self.imageLine.move(0, 20)
        self.imageLine.resize(840, 60)
        self.bt1 = QPushButton("选择图片", self)
        self.bt1.setToolTip("点击选择图片")
        self.bt1.resize(100, 60)
        self.bt1.setFont(QFont(None, 8))
        self.bt1.move(880, 20)
        self.bt1.clicked.connect(self.selectFile)
        self.bt2 = QPushButton("开始搜索", self)
        self.bt2.setToolTip("点击开始搜索")
        self.bt2.resize(100, 60)
        self.bt2.setFont(QFont(None, 8))
        self.bt2.move(1040, 20)
        self.bt2.clicked.connect(retrieval)
        self.bt3 = QPushButton("向库中添加图片", self)
        self.bt3.setToolTip("点击选择图片")
        self.bt3.resize(160, 60)
        self.bt3.setFont(QFont(None, 8))
        self.bt3.move(1200, 20)
        self.bt3.clicked.connect(addImage)
        self.lbArray = []
        self.lbTarget = QLabel(self)
        self.lbTarget.resize(200, 200)
        self.lbTarget.move(700, 100)
        self.lbTarget.show()
        for i in range(10):
            lb = QLabel(self)
            lb.resize(300, 300)
            if i<5:
                lb.move(310*i, 350)
            else:
                lb.move(310*(i-5), 660)
            lb.setText("image"+str(i))
            self.lbArray.append(lb)
    def selectFile(self):
        self.imageLine.setText(
            QFileDialog.getOpenFileName(
                self, "选择要搜索的图片", "E:/tmp/image")[0])
        self.imageLine.show()


    def showResult(self, Result, absolutePath=False):
        '''
        显示搜索前10的结果
        :param Result:  搜索结果
        :return: 无
        '''
        print(type(Result))
        resLen = len(Result)
        print("showResutl",resLen)
        image = QImage(self.imageLine.text())
        self.lbTarget.setPixmap(QPixmap.fromImage(image.scaled(200, 200, QtCore.Qt.KeepAspectRatio)))
        self.lbTarget.show()
        for item, i in zip(Result, range(resLen)):
            print(item)
            image = QImage(os.path.join('E:/tmp/image/', item[0]) if not absolutePath else item[0])
            #print("getImage",item[0])
            iLabel = self.lbArray[i]
            iLabel.clear()
            iLabel.update()
            iLabel.setPixmap(QPixmap.fromImage(image.scaled(300, 300, QtCore.Qt.KeepAspectRatio)))
            iLabel.show()

    def onApplicationExit(self, exit):
        self.exit = exit
    def closeEvent(self, QCloseEvent):
        self.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mu = user()
    mu.show()
    sys.exit(app.exec_())