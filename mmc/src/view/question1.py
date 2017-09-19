import math
import numpy as np


Distance = [
    [5732, 36601, 74681, 55505, 50424, 88242, 75407, 120000],
    [36061, 25442, 38658, 26473, 37971, 66048, 45267, 107918],
    [74681, 38658, 24684, 30445, 54379, 57300, 29720, 76935],
    [55505, 26473, 30445, 9468, 23990, 40034, 19979, 68989],
    [50424, 37971, 54379, 23990, 18445, 38620, 35717, 70555],
    [88242, 66048, 57300, 40034, 38620, 0, 27657, 31973],
    [75407, 45267, 29720, 19979, 35717, 27657, 0, 51455],
    [120000, 107918, 76935, 68989, 70555, 31973, 51455, 0]
]

unArrivedPort = [0, 1, 2, 3, 4, 5, 6]
bestPath = []
path = []
def findNextPath(mcurrentPath, pathLeft, distanceLeft, mareaLeft):
    '''
    寻找下一跳路径
    :param currentPath:
    :param pathLeft:
    :param distanceLeft:
    :return:
    '''
    currentPath = mcurrentPath.copy()
    areaLeft = mareaLeft.copy()
    for item in pathLeft:
        lastPath = currentPath[-1]
        row1 = Distance[lastPath[0]]
        row2 = Distance[item]
        #该路径已经拍摄完毕
        if areaLeft[item]<=0:
            if item == len(pathLeft)-1:
                msrow =currentPath[-1]
                if(msrow[0]!=7):
                    currentPath.append((7, 0))
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
        if row1[item] + row2[7]+ areaLeft[item] <= distanceLeft:
            # 将该路径保存
            currentPath.append((item, areaLeft[item]))
            # 减少剩余飞行航程
            distanceLeft -= (row1[item] + row2[item])
            # 在剩余区域集合中删除已经到达区域
            areaLeft[item] = 0
            findNextPath(currentPath, pathLeft, distanceLeft, areaLeft)
            # if item == 7:
            #     currentScan = 0
            #     for item in currentPath:
            #         currentScan += item[1]
            #     if len(bestPath) <= 0:
            #         bestPath.append((currentPath, currentScan))
            #     else:
            #         tPath = bestPath[0]
            #         tScan = tPath[1]
            #         if tScan<currentScan:
            #             bestPath.clear()
            #             bestPath.append((currentPath, currentScan))
            # else:
            #     findNextPath(currentPath, pathLeft, distanceLeft, areaLeft)
        #到下一区域返回还有的剩时间,但不够全部拍摄
        elif row1[item] + row2[7] < distanceLeft:
            scanDis = distanceLeft - row1[item] - row2[7]
            # areaLeft[item] -= scanDis
            tCurrentPath = currentPath.copy()
            tCurrentPath.append((item, scanDis))
            tCurrentPath.append((7, 0))
            currentScan = 0
            for titemPath in tCurrentPath:
                currentScan += titemPath[1]
            if len(bestPath) <= 0:
                bestPath.append(tCurrentPath)
            else:
                tPath = bestPath[0]
                tScan = findDistance(tPath)
                if tScan < currentScan:
                    bestPath.clear()
                    bestPath.append(tCurrentPath)
            continue
            #返回
        else:
            # 到下一区域不够，或者刚好，则不去
            # 最后一点，则回程
            if item == 6:
                currentPath.append((7, 0))
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
    if length<1:
        return
    res = 0
    for item in path:
        res += item[1]
    return res


def listMinusH(list1, list2):
    for m in list2:
        for n in list1:
            if n==m:
                list1.remove(n)
                break
    return list1

def run():
    global unArrivedPort
    scanLeft = []
    for i in range(0, 7):
        row = Distance[i]
        scanLeft.append(row[i])
    scanLeftSum = sum(scanLeft)
    lastDis = scanLeftSum+1
    while lastDis > scanLeftSum:
        lastDis = scanLeftSum
        mPath =[]
        distanceLeft = 240000
        currentPath = [(7, 0)]
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
        if len(fitPath)>0:
            path.append(fitPath)
            for item in fitPath:
                if item[0] == 7:
                    continue
                scanLeft[item[0]] -= item[1]
        print(scanLeft)
        scanLeftSum = sum(scanLeft)

def changeList(l):
    l += 2

if __name__ == '__main__':
    run()
    print(path)
    # scanLeft = []
    # for i in range(0, 3):
    #     row = Distance[i]
    #     scanLeft.append(row[i])
    # findNextPath([(7,0)],[0,1,2], 2400000,scanLeft)
    # print(bestPath)
