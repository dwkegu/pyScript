from matplotlib import pyplot as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mmc.asserts.ADataProvider as dp

mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

pointPath = 'e:/'
if __name__ == "__main__":
    figure = mpl.figure()
    # ax = Axes3D(figure)
    dataProvider = dp.DataProvider()
    # X = np.arange(0,dataProvider.getX()*38.2,38.2)
    # Y = np.arange(0,dataProvider.getY()*38.2,38.2)
    # X, Y = np.meshgrid(X, Y)
    # Z = dataProvider.getData()
    # ax.set_aspect('equal','datalim')
    # ax.plot_surface(X,Y,Z,cmap='rainbow')
    blue = 0
    red = 0
    green = 0.2
    data = dataProvider.getData1()
    x = []
    y = []
    x3 = []
    y3 = []
    colors = []
    for i in range(0, dataProvider.x):
        for j in range(0, dataProvider.y):
            if data[i*4,j*4]>3000:
                x.append(i)
                y.append(j)
                # colors.append(data[i,j]-4150)
    ax = figure.add_subplot(111)
    ax.scatter(x3, y3, s=1, c='aqua')
    ax.scatter(x, y, s=1, c='green')
    ax.scatter([110000/38.2], [0],s=10, c='red')
    ax.text(110000/38.2, 0, 'H')
    pointsX = [30300/38.2, 66000/38.2, 98400/38.2, 73700/38.2, 57900/38.2, 86800/38.2, 93600/38.2]
    pointsY = [89800/38.2, 84700/38.2, 76700/38.2, 61000/38.2, 47600/38.2, 22000/38.2, 48800/38.2]
    points = []
    for x,y in zip(pointsX, pointsY):
        points.append((x,y))
    ax.scatter(pointsX, pointsY, s=10, c='red')
    print(points)
    for point in points:
        ax.add_artist(mpl.Circle(point,radius=5000/38.2, fill=False, color='red'))
    ax.text(pointsX[0], pointsY[0], 'A')
    ax.text(pointsX[1], pointsY[1], 'B')
    ax.text(pointsX[2], pointsY[2], 'C')
    ax.text(pointsX[3], pointsY[3], 'D')
    ax.text(pointsX[4], pointsY[4], 'E')
    ax.text(pointsX[5], pointsY[5], 'F')
    ax.text(pointsX[6], pointsY[6], 'G')
    mpl.axis([-100,3000,-100,3000])
    mpl.title(u'灾区地形')
    mpl.show()

