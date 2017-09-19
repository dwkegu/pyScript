from mmc.asserts.ADataProvider import DataProvider
import pyclust
import numpy as np
from matplotlib import pyplot as mpl
import math


mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
class Cluster:
    def __init__(self, n_cluster):
        self.km = pyclust.KMeans(n_clusters=n_cluster,n_trials=100, max_iter=2000)
        self.kkm = pyclust.KernelKMeans(n_cluster,kernel='rbf',params={'gamma':0.5})

    def clust(self, x):
        self.km.fit(x)
        return self.km.labels_


if __name__ == "__main__":
    dp = DataProvider()
    data = dp.getData1()
    xNum = dp.x
    yNum = dp.y
    x = []
    unArrival = []
    pointsCount = []
    for i in range(0,20):
        pointsCount.append(0)
    for i in range(0,xNum):
        for j in range(0,yNum):
            if(data[i,j]>4000):
                unArrival.append((i,j))
            elif(int(i%100)==0 and int(j%100)==0 and data[i,j]<=3000):
                x.append([i,j])
    mCluster = Cluster(20)
    x = np.asarray(x)
    mCluster.clust(x)
    cals = []
    for i in range(0,20):
        cals.append([])
    lables = mCluster.km.labels_
    centers = mCluster.km.centers_
    print(centers)
    #海拔和
    totalHeight = []
    for i in range(0,20):
        totalHeight.append(0)
    for i in range(0,len(lables)):
        pointsCount[lables[i]] += 1
        totalHeight[lables[i]] += data[x[i,0],x[i,1]]
        cals[lables[i]].append(x[i, :])
    mpl.figure()
    unArrival = np.asarray(unArrival)
    mpl.scatter(unArrival[:, 0], unArrival[:, 1], s=1, c='green')
    cs = [i for i in range(0, 100, 5)]
    cmap = mpl.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    for i in range(0,20):
        ta = np.asarray(cals[i])
        mpl.scatter(ta[:,0],ta[:,1],s=1, c=colors[i])
        mpl.scatter(centers[i,0],centers[i,1],s=20, c=colors[i])
    mpl.scatter([110000/38.2,],[0],s=10,c='red')
    mpl.title(u'3000米以下分类图')
    #等效面积
    areaCount = []
    # 平均海拔
    averageHeight = []
    equalDis = []
    for i in range(0,len(pointsCount)):
        areaCount.append(int(pointsCount[i]*10000*38.2*38.2))
        averageHeight.append(int(totalHeight[i]/pointsCount[i]))
        equalDis.append(int(areaCount[i]/((4000-averageHeight[i])/math.sqrt(3))))
    print(areaCount)
    print(averageHeight)
    print(equalDis)
    mpl.show()

