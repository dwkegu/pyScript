
a = [2658, 2357, 2051, 2696, 2212, 2737, 2180, 2565, 2352, 2585, 2360, 2418, 2395, 2667, 2783, 2559, 2671, 2657, 2616, 2565]
b = [233478400, 350217600, 233478400, 102146800, 248070800, 189701200, 262663200, 218886000, 160516400, 145924000, 131331600, 233478400, 248070800, 291848000, 131331600, 145924000, 175108800, 116739200, 175108800, 160516400]
c = [int(i/j) for i,j in zip(b,a)]
print(c)

