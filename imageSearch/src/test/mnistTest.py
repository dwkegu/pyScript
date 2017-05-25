from imageSearch.src import myEvalInceptionV4 as engine
import os
import csv
from imageSearch.src.PreProcessing.dataSetPreProcessing import mnistDir
from matplotlib import pyplot as plot
from imageSearch.src.compare import comHashCode
import re
import numpy as np
import io

def mnistEval(filepath):
    net = engine.ImageNet()
    imagesDir = os.listdir(filepath)
    resultCount = 0
    correct = 0
    allResult = []
    with open(os.path.join(mnistDir+'/', 'hashCode.csv'), 'r') as f:
        hashCode = csv.reader(f)
        for numItem in imagesDir:
            if filepath.endswith('/'):
                numDir = os.path.join(filepath, numItem)
            else:
                numDir = os.path.join(filepath+'/', numItem)
            images = os.listdir(numDir)
            num = int(numItem)
            for image in images:
                resultCount +=10
                imagefile = os.path.join(numDir+'/', image)
                res = net.test(batch_size=1, image_filename = imagefile, operation=0)
                hc1 = res['hc1']
                f.seek(0, io.SEEK_SET)
                correct += Retrieval(hashCode, hc1, num)
                if resultCount % 1000 == 0:
                    break
            allResult.append(correct)
    for i in range(10):
        allResult[i] /= (100*i+100)
    x_value = np.arange(100, 1100, 100)
    print(allResult)
    plot.plot(x_value, allResult)
    plot.show()

def Retrieval(hashCode_csv, hc, num):
    num1 = 0
    min1 = []
    correct = 0
    for row in hashCode_csv:
        filepath = row[0]
        hc1 = row[1]
        dist = comHashCode.compareHashCode(hc, hc1[1:-1].split(','))
        if num1 < 10:
            num1 += 1
            min1.append((filepath, dist))
        else:
            min1.sort(key=lambda k: k[1])
            item = min1[9]
            if dist < item[1]:
                min1.pop()
                min1.append((row[0], dist))
    min1.sort(key=lambda k: k[1])
    min1 = min1[0:1]
    for item in min1:
        tnum = int(re.match(r'^.*\/([0-9])\/.*\.(?:png|jpg)$', item[0]).groups()[0])
        if tnum == num:
            correct+=1
    return correct

if __name__ == '__main__':

    result = [0.94, 0.96, 0.8533333333333334, 0.84, 0.84, 0.805, 0.7971428571428572, 0.7825, 0.7922222222222223, 0.779]
    count = 0
    x = np.arange(100, 1100, 100)
    #fig = pylab.plot.figure()
    plot.plot(x, result)
    #fig.suptitle('mnist测试', fontsize=20)
    plot.xlabel('test epoch', fontsize=18)
    plot.ylabel('Accuracy', fontsize=16)
    plot.show()
    '''
    mnistEval('E:/tmp/dataSet/mnist_png/testing')
    '''