import random

import numpy as np

from red import Red
from graficas import *

SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2
SEED = 42

def cargar_datos(filename):
    x = []
    y = []
    with open(filename) as fp:
        for line in fp:
            s = line.split(",")
            rasgos = np.array(map(float, s[:-1])).reshape(4,1)
            x.append(rasgos)

            if s[-1] == "Iris-setosa\n":
                y.append(SETOSA)
            elif s[-1] == "Iris-versicolor\n":
                y.append(VERSICOLOR)
            elif s[-1] == "Iris-virginica\n":
                y.append(VIRGINICA)

    return zip(x, y)


def bin_data(data):
    x, y = map(list, zip(*data))
    for i in range(len(y)):
        if y[i] != 0:
            y[i] = 1
    return zip(x, y)


def split(data, percent):
    p = percent*len(data)/100
    return (data[:p], data[p:])

data = cargar_datos("iris.data")
b_data = bin_data(data)

random.seed(SEED)
random.shuffle(data)
random.shuffle(b_data)

redes = [4,5,6,7,8,9,10]
splits = [50,60,70,80,90]

################################################################################

file = open("resultados/e3/resultados_bin.csv", 'w')
for i, r in enumerate(redes):
    costos = []
    for s in splits:
        train_data, test_data = split(b_data, s)
        red = Red([4,r,2])
        costo = red.SGD(train_data, 600, len(train_data)/4, 0.015)
        costos.append(costo)
        acc = red.accuracy(test_data, 4)
        file.write("{0:.2f}%".format(round(acc,2)))
        file.write(",")
    file.write("\n")
    convergencia_por_conjunto(costos, splits, "plots/e3/red"+str(i+4)+"_bin.png", False, i+4)
file.close()

################################################################################
################################################################################

file = open("resultados/e3/resultados_clas.csv", 'w')
for i, r in enumerate(redes):
    costos = []
    for s in splits:
        train_data, test_data = split(data, s)
        red = Red([4,r,3])
        costo = red.SGD(train_data, 600, len(train_data)/4, 0.015)
        costos.append(costo)
        acc = red.accuracy(test_data, 4)
        file.write("{0:.2f}%".format(round(acc,2)))
        file.write(",")
    file.write("\n")
    convergencia_por_conjunto(costos, splits, "plots/e3/red"+str(i+4)+"_clas.png", False, i+4)
file.close()

################################################################################
