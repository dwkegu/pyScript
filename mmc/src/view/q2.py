import numpy as np
import math
from mmc.src.view import cluster
from matplotlib import pyplot as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

centers = [[450., 1393.75],
           [1529.16666667, 2295.83333333],
           [1950, 2481.25],
           [71.42857143, 2200],
           [1764.70588235, 2794.11764706],
           [484.61538462, 753.84615385],
           [2361.11111111, 2805.55555556],
           [646.66666667, 1860],
           [400, 2818.18181818],
           [80., 2750.],
           [2266.66666667, 2311.11111111],
           [1237.5, 2750.],
           [1158.82352941, 1935.29411765],
           [1620., 1505.],
           [2555.55555556, 2144.44444444],
           [340., 2490.],
           [1433.33333333, 1133.33333333],
           [2587.5, 2487.5],
           [2083.33333333, 2066.66666667],
           [836.36363636, 2681.81818182],
           # [110000 / 38.2, 0],
           [110000 / 38.2, 55000 / 38.2]
           ]
HC = [[450., 1393.75],
      [71.42857143, 2200],
      [484.61538462, 753.84615385],
      [646.66666667, 1860],
      [80., 2750.],
      [1158.82352941, 1935.29411765],
      [1620., 1505.],
      [340., 2490.],
      [1433.33333333, 1133.33333333],
      [110000 / 38.2, 0], ]
JC = [[1529.16666667, 2295.83333333],
      [1950, 2481.25],
      [1764.70588235, 2794.11764706],
      [2361.11111111, 2805.55555556],
      [400, 2818.18181818],
      [2266.66666667, 2311.11111111],
      [1237.5, 2750.],
      [2555.55555556, 2144.44444444],
      [2587.5, 2487.5],
      [2083.33333333, 2066.66666667],
      [836.36363636, 2681.81818182],
      [110000 / 38.2, 55000 / 38.2]
      ]
centers = np.asarray(centers)
mDistance = []
# lookDis = [301338, 369199, 207489, 135677, 240308, 260152, 249970, 264196, 168703, 178620, 138703, 255623, 267707,
#            379216, 186912, 175397, 228214, 150557, 219145, 193743, 0]
lookDis = [87839, 148586, 113836, 37888, 112147, 69309, 120487, 85335, 68246, 56450, 55648, 96558, 103578, 109429,
           47190, 57023, 65559, 43936, 66937, 62579,0]
for i in range(0, 21):
    tRow = []
    for j in range(0, 21):
        if i != j:
            tRow.append(
                int(math.sqrt((centers[j, 0] - centers[i, 0]) ** 2 + (centers[j, 1] - centers[i, 1]) ** 2) * 38.2))
        else:
            tRow.append(int(lookDis[i]))
    mDistance.append(tRow)

bestPath = []
path = []


def findNextPath(mcurrentPath, pathLeft, distanceLeft, mareaLeft):
    '''
    寻找下一跳路径
    :param mcurrentPath:
    :param pathLeft:
    :param distanceLeft:
    :return:
    '''
    currentPath = mcurrentPath.copy()
    areaLeft = mareaLeft.copy()
    for item in pathLeft:
        lastPath = currentPath[-1]
        row1 = mDistance[lastPath[0]]
        row2 = mDistance[item]
        # 该路径已经拍摄完毕
        if areaLeft[item] <= 0:
            if item == pathLeft[-1]:
                currentPath.append((startPoint, 0))
                currentScan = 0
                for titemPath in currentPath:
                    currentScan += titemPath[1]
                if len(bestPath) <= 0:
                    bestPath.append(currentPath)
                else:
                    tPath = bestPath[0]
                    tScan = findDistance(tPath)
                    if tScan < currentScan:
                        bestPath.clear()
                        bestPath.append(currentPath)
                return
            else:
                continue
        # 剩余航程支持该路径探测剩余全部并回程还有的多
        if row1[item] + row2[startPoint] + areaLeft[item] <= distanceLeft:
            # 将该路径保存
            currentPath.append((item, areaLeft[item]))
            # 减少剩余飞行航程
            distanceLeft -= (row1[item] + row2[item])
            # 在剩余区域集合中删除已经到达区域
            areaLeft[item] = 0
            findNextPath(currentPath, pathLeft, distanceLeft, areaLeft)
        # 到下一区域返回还有的剩时间,但不够全部拍摄
        elif row1[item] + row2[startPoint] < distanceLeft:
            scanDis = distanceLeft - row1[item] - row2[startPoint]
            _currentPath = currentPath.copy()
            _currentPath.append((item, scanDis))
            # areaLeft[item] -= scanDis
            _currentPath.append((startPoint, 0))
            currentScan = 0
            for titemPath in _currentPath:
                currentScan += titemPath[1]
            if len(bestPath) <= 0:
                bestPath.append(_currentPath)
            else:
                tPath = bestPath[0]
                tScan = findDistance(tPath)
                if tScan < currentScan:
                    bestPath.clear()
                    bestPath.append(_currentPath)
            continue
            # 返回
        else:
            # 到下一区域不够，或者刚好，则不去
            # 最后一点，则回程
            if item == pathLeft[-1]:
                currentPath.append((20, 0))
                currentScan = 0
                for item in currentPath:
                    currentScan += item[1]
                if len(bestPath) <= 0:
                    bestPath.append((currentPath))
                else:
                    tPath = bestPath[0]
                    tScan = findDistance(tPath)
                    if tScan < currentScan:
                        bestPath.clear()
                        bestPath.append(currentPath)
            # 若未到最后一站则继续
            else:
                continue


def findDistance(path):
    length = len(path)
    if length < 1:
        return 0
    res = 0
    for item in path:
        res += item[1]
    return res


def findPathTime(path):
    length = len(path)
    if length < 1:
        return 0
    res = 0
    start = path[0]
    for i in range(1, length):
        item = path[i]
        row1 = mDistance[start[0]]
        res += (row1[item[0]] + item[1])
        start = item
    return res / 60000


def listMinusH(list1, list2):
    for m in list2:
        for n in list1:
            if n == m:
                list1.remove(n)
                break
    return list1


# unArrivedPort = [0, 3, 5, 7, 12, 13, 16]
unArrivedPort = [1, 2, 4, 6, 8, 9, 10, 11, 14, 15, 17, 18, 19]
startPoint = 20
# for i in range(0,20):
#     unArrivedPort.append(i)
if __name__ == "__main__":
    scanLeft = []
    for i in range(0, 20):
        row = mDistance[i]
        scanLeft.append(row[i])
    scanLeftSum = sum(scanLeft)
    # print(scanLeft)
    lastDis = scanLeftSum + 1
    while lastDis > scanLeftSum:
        lastDis = scanLeftSum
        mPath = []
        distanceLeft = 260851
        currentPath = [(startPoint, 0)]
        pathLeft = unArrivedPort
        bestPath.clear()
        findNextPath(currentPath, pathLeft, distanceLeft, scanLeft)
        if len(bestPath) > 0:
            dis = []
            for item in bestPath:
                dis.append(findDistance(item))
            maxIndex = 0
            maxDis = dis[0]
            for j in range(0, len(dis)):
                if dis[j] > maxDis:
                    maxDis = dis[j]
                    maxIndex = j
            mPath.append(bestPath[maxIndex])
        pathLength = 0
        fitPath = []
        for itemPath in mPath:
            if findDistance(itemPath) > pathLength:
                fitPath = itemPath
                pathLength = findDistance(itemPath)
        if len(fitPath) > 0:
            path.append(fitPath)
            for item in fitPath:
                if item[0] == 20:
                    continue
                scanLeft[item[0]] -= item[1]
        # print(scanLeft)
        scanLeftSum = sum(scanLeft)
    for item in path:
        print(item, findPathTime(item))
    print(scanLeft)
    print(len(path))
