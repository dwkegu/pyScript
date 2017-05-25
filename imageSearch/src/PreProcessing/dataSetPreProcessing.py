import os
import csv

mnistDir = 'E:/tmp/dataSet/mnist_png/base'
cifarDir = 'E:/tmp/dataSet/cifar10/base'
flowersDir = 'E:/tmp/dataSet/flower_photos/base'
sep = '/'
hashFileName = 'hashCode.csv'
def mnistProcess(filepath, hc1, hc2):
    '''
    对mnist数据集进行哈希码提取
    :param filepath: 原图片文件位置
    :param hc1: 图片1536维哈希码
    :param hc2: 图片分类哈希码
    '''
    if not os.path.exists(mnistDir):
        os.mkdir(mnistDir)
    if not os.path.exists(os.path.join(mnistDir+sep, hashFileName)):
        with open(os.path.join(mnistDir+sep, hashFileName), 'w') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)
    else:
        with open(os.path.join(mnistDir+sep, hashFileName), 'a') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)

def cifar10Process(filepath, hc1, hc2):
    '''
    对cifar数据集进行哈希码提取
    :param filepath: 原图片文件位置
    :param hc1: 图片1536维哈希码
    :param hc2: 图片分类哈希码
    '''
    if not os.path.exists(cifarDir):
        os.mkdir(cifarDir)
    if not os.path.exists(os.path.join(cifarDir+sep, hashFileName)):
        with open(os.path.join(cifarDir+sep, hashFileName), mode='w', newline='') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)
    else:
        with open(os.path.join(cifarDir+sep, hashFileName), mode='a', newline='') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)

def flowersProcess(filepath, hc1, hc2):
    '''
    对flowers数据集进行哈希码提取
    :param filepath: 原图片文件位置
    :param hc1: 图片1536维哈希码
    :param hc2: 图片分类哈希码
    '''
    if not os.path.exists(flowersDir):
        os.mkdir(flowersDir)
    if not os.path.exists(os.path.join(flowersDir+sep, hashFileName)):
        with open(os.path.join(flowersDir+sep, hashFileName), mode='w', newline='') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)
    else:
        with open(os.path.join(flowersDir+sep, hashFileName), mode='a', newline='') as f:
            f_csv = csv.writer(f)
            row = [filepath, hc1, hc2]
            f_csv.writerow(row)