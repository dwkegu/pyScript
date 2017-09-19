from mmc.asserts.ADataProvider import DataProvider
import math

#搜索区域的坐标
pointsX = [30300/38.2, 66000/38.2, 98400/38.2, 73700/38.2, 57900/38.2, 86800/38.2, 93600/38.2]
pointsY = [89800/38.2, 84700/38.2, 76700/38.2, 61000/38.2, 47600/38.2, 22000/38.2, 48800/38.2]
dp = DataProvider()
data = dp.getData1()
x = dp.x
y = dp.y
pointsCount = [0,0,0,0,0,0,0]
totalHeight = [0,0,0,0,0,0,0]
#搜索区域的半径
MAX_POINT_DISTANCE = 130
#搜索圆的面积
CIRCLE_AREA = math.pi*10000**2/4
#搜索圆内的最大点数
CIRCLE_POINTS = int(53821)
for i in range(0,x):
    for j in range(0,y):
        if data[i,j] <= 3000:
            for k in range(0,7):
                dis = math.sqrt(math.pow(pointsX[k]-i, 2)+math.pow(pointsY[k]-j, 2))
                if dis <= MAX_POINT_DISTANCE:
                    totalHeight[k] += data[i,j]
                    pointsCount[k] += 1
averageHeight = []
for item in range(0, 7):
    if pointsCount[item]==0:
        averageHeight.append(0)
        continue
    averageHeight.append(str(int((4200-(totalHeight[item]/pointsCount[item]))/math.sqrt(3))))
xu = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for key,value in zip(xu,pointsCount):
    print(key+":"+str(value))
areas = []
for item in pointsCount:
    areas.append(int(item/CIRCLE_POINTS*CIRCLE_AREA))
for key,value in zip(xu, areas):
    print(key+":"+str(value))
for key, value in zip(xu, averageHeight):
    print(key+":"+str(value))