import os
import csv

def compareHashCode(hashCode1, hashCode2):
    distance = 0.0
    for item1, item2 in zip(hashCode1, hashCode2):
        distance += abs(int(item1)-int(item2))
    #print(distance)
    return distance
