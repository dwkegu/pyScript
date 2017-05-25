from imageSearch.src import myEvalInceptionV4 as engin
import numpy as np
import pylab
import os
from imageSearch.src.compare import comHashCode
from imageSearch.src.userInterface import baseUI
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import csv
from io import SEEK_SET

DEBUG = False
def test2(image):
    mnistDir = 'E:/tmp/dataSet/mnist_png/base/'
    cifar10Dir = 'E:/tmp/dataSet/cifar10/base/'
    flowersDir = 'E:/tmp/dataSet/flower_photos/base/'
    diyImage = 'D:/tmp/data/image2/imageAll'
    global net
    if net == None:
        net = engin.ImageNet()
    res = net.test(batch_size=1, image_class_dir='E:/tmp/class/',
               image_filename=image, operation=3)
def test3():
    net = engin.ImageNet()
    flowersDir = 'E:/tmp/dataSet/flower_photos/base/'
    dirs = os.listdir(flowersDir)
    for i in dirs:
        if i.endswith('.csv'):
            continue
        imagePath = os.path.join(flowersDir, str(i))
        net.test(batch_size=1, image_filename=imagePath, operation=1)
def test1():
    global net
    if net==None:
        net = engin.ImageNet()
    x_value = np.arange(0, 1001, 1)
    #print(x_value.shape)
    testFileName = [#'/tmp/psf/ImageSearch/1493111854.0703988fengshu_he_fengye-006.jpg',
                    #'/tmp/psf/ImageSearch/1493111854.0703988fengshu_he_fengye-006_rotation.jpg',
                    #'/tmp/psf/ImageSearch/1493111854.0703988fengshu_he_fengye-006color.jpg',
                    #'/tmp/psf/ImageSearch/1493109416.9236438baishouwan-005.jpg',
                    #'/tmp/psf/ImageSearch/1493111854.0703988fengshu_he_fengye-006cut.jpg',
                    #'/tmp/psf/ImageSearch/1493111854.0703988fengshu_he_fengye-006scale.jpg',
                    'E:/tmp/psf/test.jpg'
    ]
    hc = []
    for item in testFileName:
        res = net.test(batch_size=1, image_class_dir='f:/tmp/psf/classImage',
                       image_filename=item, operation= 0)
        #print(len(res["PreLogitsFlatten"]))
        y_value = np.reshape(np.asarray(res["class"]), [-1])
        hc.append(y_value)
        # y_value = np.reshape(np.asarray(res["hc1"]),[-1])
        # y_value.sort(0)
        #print(y_value)
        #print(y_value.shape)

        pylab.plot(x_value, y_value, linewidth = 0.5)
    pylab.title("classification output")
    pylab.show()
    # print(comHashCode.compareHashCode(hc[0], hc[1]))
    ''' 比较不同图片的哈希距离
    print(comHashCode.compareHashCode(hc[0], hc[2]))
    print(comHashCode.compareHashCode(hc[1], hc[2]))
    '''

searchImageFilename =''
imageHashPath = 'E:/tmp/image/hashCode.csv'
mui = None
net = None
imageHash = None
imageHash_csv = None

def startRetrieval():
    global net
    global mui
    global searchImageFilename
    global imageHash
    global imageHashPath
    global imageHash_csv
    searchImageFilename = mui.imageLine.text()
    print(searchImageFilename)
    code = net.test(batch_size=1, image_class_dir='f:/tmp/psf/classImage',
                   image_filename=searchImageFilename, operation=0)
    # print(len(res["PreLogitsFlatten"]))
    if imageHash is None:
        imageHash = open(imageHashPath, 'r')
        imageHash_csv = csv.reader(imageHash)
    y_value1 = code["hc2"]
    y_value0 = code["hc1"]
    max1_10 = 1536
    max2_10 = 1001
    min1 = []
    num1 = 0
    print("start to get 10 similar images")
    for row in imageHash_csv:
        #print(row)
        row1 = str(row[1])
        dis1 = comHashCode.compareHashCode(y_value0, list(row1[1:-1].split(',')))
        if num1 < 10:
            num1 += 1
            min1.append((row[0], dis1))
        else:
            min1.sort(key=lambda k:k[1])
            item = min1[9]
            if dis1<item[1]:
                min1.pop()
                min1.append((row[0], dis1))
    min1.sort(key=lambda k:k[1])
    min1 = min1[:10]
    imageHash.seek(0, SEEK_SET)
    print("finish compare")
    mui.showResult(min1, absolutePath=False)
def AcrossB(a, b):
    for item in a:
        for i in b:
            #print(type(item), type(i))
            if item==i:
                return True
            else:
                continue
    return False
def startRetrieval2():
    global net
    global mui
    global searchImageFilename
    global imageHash
    global imageHashPath
    global imageHash_csv
    searchImageFilename = mui.imageLine.text()
    print(searchImageFilename)
    code = net.test(batch_size=1, image_class_dir='f:/tmp/psf/classImage',
                    image_filename=searchImageFilename, operation=0)
    # print(len(res["PreLogitsFlatten"]))
    if imageHash is None:
        imageHash = open(imageHashPath, 'r')
        imageHash_csv = csv.reader(imageHash)
    y_value1 = code["topK"]
    y_value0 = code["hc1"]
    print(y_value1)
    print(y_value0)
    max1_10 = 1536
    max2_10 = 1001
    min1 = []
    num1 = 0
    print("start to get 10 similar images")
    for row in imageHash_csv:
        #print(row)
        row1 = str(row[1])
        topKind = row[2]
        topKind = map(int, topKind[1:-1].split(','))
       # print(y_value1,topKind)
        if not AcrossB(y_value1, topKind):
            continue
        dis1 = comHashCode.compareHashCode(y_value0, list(row1[1:-1].split(',')))
        if num1 < 10:
            num1 += 1
            min1.append((row[0], dis1))
        else:
            min1.sort(key=lambda k:k[1])
            item = min1[9]
            if dis1<item[1]:
                min1.pop()
                min1.append((row[0], dis1))
    min1.sort(key=lambda k:k[1])
    if len(min1) >10:
        min1 = min1[:10]
    imageHash.seek(0, SEEK_SET)
    print("finish compare")
    mui.showResult(min1, absolutePath=False)

def addImagetoBase():
    global imageHash
    image = QFileDialog.getOpenFileName(mui, "选择要搜索的图片", "E:/tmp/image")[0]
    if imageHash!=None:
        imageHash.close()
        imageHash=None
    print(image)
    test2(image)


def appExit():
    global imageHash
    if imageHash is not None:
        imageHash.close()
        imageHash = None
    if net is not None:
        net.close()

if __name__ == '__main__':
    # test1()
    "InceptionV4/Logits/PreLogitsFlatten/Reshape"
    "InceptionV4/Logits/Predictions"

    net = engin.ImageNet()
    mapp = QApplication(sys.argv)
    mui = baseUI.user(startRetrieval2,addImagetoBase)
    mui.show()
    mui.onApplicationExit(appExit)
    sys.exit(mapp.exec_())