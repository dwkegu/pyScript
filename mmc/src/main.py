import matplotlib as mpl
from gaft import GAEngine
from math import sin, cos
import random
import numpy as np
from gaft.components import GAIndividual, GAPopulation
from  gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from gaft.analysis.fitness_store import FitnessStoreAnalysis
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from mmc.asserts.DataProvider import DataProvider
from mmc.src.CustomIndividual import CustomIndividual

data = DataProvider()
aircraftsData = data.getAircraftsData()
schedulesData = data.getSchulesData()
variable_ranges = []
for i in range(0, 97):
    variable_ranges.append((0, 30))
    variable_ranges.append((0, 16))
indv_temp = CustomIndividual(schedulesData[2], aircraftsData[2], ranges=variable_ranges, encoding='binary', eps=0.5)
population = GAPopulation(indv_template=indv_temp, size=50).init()
for indiv in population.individuals:
    for item in indiv.gene_indices:
        print(item)
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.4, pe=0.2)
mutation = FlipBitMutation(pm=0.4)
mEngine = GAEngine(population=population, selection=selection,
                   crossover=crossover, mutation=mutation,
                   analysis=[FitnessStoreAnalysis])


@mEngine.fitness_register
def fitness(indv):
    x = indv.variants
    fitness = float(-1)
    # 约束条件 惩罚项
    time = 0
    plane = 0
    ovsCheck1 = {}
    ovsCheck2 = {}
    for i, item in zip(range(0, 194), x):
        item = int(item)
        if int(i % 2) == 0:
            # 取消起飞
            if item == 0:
                fitness -= 10000
            else:
                # 检查OVS起飞降数量限制
                if checkOVS(int(i / 2)) == 1:
                    # 在OVS起飞
                    currentFlight = sData[int(i / 2)]
                    St = int(currentFlight[1]) + 10 * 60 * (item - 1)
                    St /= (5 * 60)
                    if str(St) in ovsCheck1.keys():
                        ovsCheck1[str(St)] += 1
                    else:
                        ovsCheck1[str(St)] = 1
                elif checkOVS(int(i / 2)) == 2:
                    # 在OVS降落
                    currentFlight = sData[int(i / 2)]
                    St = int(currentFlight[2]) + 10 * 60 * (item - 1)
                    St /= (5 * 60)
                    if str(St) in ovsCheck2.keys():
                        ovsCheck2[str(St)] += 1
                    else:
                        ovsCheck2[str(St)] = 1
                time = item
        else:
            # 检查飞机是否可用
            if checkPlane(int(i/2), time, item, x):
                fitness += (-10 * 60 * (time - 1))
            else:
                fitness -= 10000
    for key in ovsCheck1:
        if ovsCheck1[key] > 5:
            fitness -= 10000
    for key in ovsCheck2:
        if ovsCheck2[key] > 5:
            fitness -= 10000
    return fitness


def checkPlane(f, t, p, x):
    # 检查是否在同一机场
    sData = schedulesData[2]
    pData = aircraftsData[2]
    dTime = 0
    #目标航班信息
    aimFlightInfo = sData[f]
    #该航班起始机场
    flightS = str(aimFlightInfo[3]).strip().lower()
    #该航班实际起飞时间
    t = int(aimFlightInfo[1])+10*60*(t-1)
    #检查初始飞机位置和间隔时间是否满足需求
    planeRow = pData[p]
    initPlaneNum = 0
    if str(planeRow[4]).strip().lower()== str(aimFlightInfo[3]).strip().lower():
        initPlaneNum = 1
    for i, item in zip(range(0, 97), x):
        item = int(item)
        if int(i%2)==0:
            dTime = 10*60*(item-1)
        #非当前航班但使用相同型号飞机
        elif int(i/2)!=int(f) and item == p:
            flightInfo = sData[int(i/2)]
            st = int(flightInfo[1])+dTime
            #某趟航班采用相同飞机在这之前起飞了
            if st<t:
                #起点机场相同
                if flightS == str(flightInfo[3]).strip().lower():
                    initPlaneNum = initPlaneNum -1
                #该航班终点机场与目标航班起点机场相同且满足间隔要求
                elif flightS == str(flightInfo[4]).strip().lower() and int(flightInfo[2])-int(flightInfo[1])+st+45<t:
                    initPlaneNum = initPlaneNum +1
    # 检查是否满足飞机可用时间
    if int(planeRow[2])<=t and int(planeRow[3])>=int(aimFlightInfo[2])-int(aimFlightInfo[1])+t and initPlaneNum>0:
        return True
    return False

def checkOVS(f):
    sData = schedulesData[2]
    row = sData[f]
    startPort = str(row[3]).strip().lower()
    endPort = str(row[4]).strip().lower()
    res = 0
    if startPort == "ovs":
        res = 1
    if endPort == "ovs":
        res = 2
    return res


@mEngine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    master_only = True
    interval = 1

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation:{}, best fitness{:.3f}'.format(g, engine.fitness(best_indv))
        engine.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.variants
        y = engine.fitness(best_indv)
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        engine.logger.info(msg)


def fitness2(x):
    fitness = float(-1)
    # 约束条件 惩罚项
    time = 0
    plane = 0
    ovsCheck1 = {}
    ovsCheck2 = {}
    for i, item in zip(range(0, 194), x):
        item = int(item)
        if int(i % 2) == 0:
            # 取消起飞
            if item == 0:
                fitness -= 10000
            else:
                # 检查OVS起飞降数量限制
                if checkOVS(int(i/2)) == 1:
                    #在OVS起飞
                    currentFlight = sData[int(i / 2)]
                    St = int(currentFlight[1]) + 10 * 60 * (item - 1)
                    St /= (5*60)
                    if str(St) in ovsCheck1.keys():
                        ovsCheck1[str(St)] += 1
                    else:
                        ovsCheck1[str(St)] = 1
                elif checkOVS(int(i/2)) == 2:
                    # 在OVS降落
                    currentFlight = sData[int(i / 2)]
                    St = int(currentFlight[2]) + 10 * 60 * (item - 1)
                    St /= (5*60)
                    if str(St) in ovsCheck2.keys():
                        ovsCheck2[str(St)] += 1
                    else:
                        ovsCheck2[str(St)] = 1
                time = item
        else:
            # 检查飞机是否可用
            if checkPlane(int(i/2), time, item, x):
                fitness += (-10 * (time - 1))
            else:
                fitness -= 10000
    for key in ovsCheck1:
        if ovsCheck1[key] > 5:
            fitness -= 10000
    for key in ovsCheck2:
        if ovsCheck2[key] > 5:
            fitness -= 10000
    print(fitness)
    return fitness

if '__main__' == __name__:
    sData = schedulesData[2]
    pData = aircraftsData[2]
    mEngine.run(ng=10)
    # variable_ranges = []
    # for i in range(0, 97):
    #     variable_ranges.append((0, 30))
    #     variable_ranges.append((0, 16))
    # length = len(variable_ranges)
    # res = []
    # for i in range(0, length):
    #     res.append(1)
    #     sData = schedulesData[2]
    #     pData = aircraftsData[2]
    #     flightRow = sData[i]
    #     planeName = str(flightRow[6]).strip().lower()
    #     i = 0
    #     for row in pData:
    #         if str(row[0]).strip().lower() == planeName:
    #             res.append(i)
    #         i += 1
    # print(res)
    # fitness2(res)

