from imageSearch.src.compare import comHashCode
import csv
from imageSearch.src import myEvalInceptionV4
import numpy as np
import tensorflow as tf
import io

if __name__ == '__main__':
    file_name = '/tmp/data/flower_photos/sunflowers/200557979_a16112aac1_n.jpg'
    file_name2 = ['/tmp/data/flower_photos/daisy/hashCode.csv']
    '''
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        mainRow = next(f_csv)
        for file in file_name2:
            with open(file, 'r') as f1:
                f1_csv = csv.reader(f1)
                print(mainRow)
                for row in f1_csv:
                    for item1, item2 in zip(mainRow[1:], row[1:]):
                        print('{0}\n{1}'.format(item1,item2))
                        print(comHashCode.compareHashCode(list(item1[1:-1].split(',')), list(item2[1:-1].split(','))))
    '''
    myNet = myEvalInceptionV4.ImageNet()
    res = myNet.test(batch_size=1, image_class_dir='f:/tmp/psf/classImage',
                   image_filename=file_name, operation= 0)
    with open(file_name2[0], 'r') as f:
        f_csv = csv.reader(f)
        main_row = res[2:]
        print(main_row)
        a_min=[]
        b_min=[]
        for row in f_csv:
            row1 = str(row[1])
            row2 = str(row[2])
            a_min.append((row[0], comHashCode.compareHashCode(main_row[0], list(row1[1:-1].split(',')))))
            b_min.append((row[0], comHashCode.compareHashCode(main_row[1], list(row2[1:-1].split(',')))))
        a_min.sort(key= lambda k: k[1])
        a_k_min = a_min[0:10]
        b_min.sort(key= lambda k: k[1])
        b_k_min = b_min[0:10]
        print(a_k_min)
        print(b_k_min)