import csv
import os
from io import SEEK_SET
import numpy as np


class DataProvider:
    def __init__(self):
        module_path = os.path.dirname(__file__)
        self.filename = module_path + '/q1d1.csv'
        with open(self.filename, 'r') as f:
            self.x = len(f.readlines())
            f.seek(0, SEEK_SET)
            f_csv = csv.reader(f)
            self.y = len(next(f_csv))
            f.seek(0, SEEK_SET)
            self.data1 = np.empty([self.x, self.y], int)
            i = int(0)
            j = int(0)
            for row in f_csv:
                j = 0
                for item in row:
                    self.data1[i, j] = int(item)
                    j += 1
                i += 1
        self._distances = []
        self._distances.append([])
        self._distances.append([])

    def getData1(self):
        return self.data1

    def getData2(self):
        return self.data2


    def getX(self):
        return self.y


    def getY(self):
        return self.x


    def getDistance(self):
        return self._distances