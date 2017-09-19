import csv
import os
import numpy as np
from io import SEEK_SET

class DataProvider:
    def __init__(self):
        module_path = os.path.dirname(__file__)
        self.filename = {'aircrafts':module_path + '/Aircrafts.csv','paxinfo':module_path+'/Paxinfo.csv',
                         'schedules':module_path+'/Schedules.csv'}


    #获取飞机数据
    def getAircraftsData(self):
        '''
        获取飞机相关信息
        信息分别为：飞机尾号，飞机型号，最早可用时间，最晚可用时间，起点机场，座位数 ...
        :return: 飞机数量，每架飞机信息列数， 数据矩阵
        '''
        with open(self.filename['aircrafts'], 'r') as f:
            rowNum = len(f.readlines())
            f.seek(0, SEEK_SET)
            f_csv = csv.reader(f)
            columnNum = len(next(f_csv))
            f.seek(0, SEEK_SET)
            data = []
            for row in f_csv:
                data.append(row)
        return rowNum, columnNum, data

    #获取航班数据
    def getSchulesData(self):
        '''
        获取航班相关信息
        航班信息分别为：航班唯一编号，起飞时间，到达时间，起飞机场，到达机场，飞机型号，飞机尾号...
        :return: 航班数量，每个航班机信息列数， 数据二维数组
        '''
        with open(self.filename['schedules'], 'r') as f:
            rowNum = len(f.readlines())
            f.seek(0, SEEK_SET)
            f_csv = csv.reader(f)
            columnNum = len(next(f_csv))
            f.seek(0, SEEK_SET)
            data = []
            for row in f_csv:
                data.append(row)
        return rowNum, columnNum, data