import numpy as np

from red import Red

SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2

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
    print y
    for i in range(len(y)):
        if y[i] != 0:
            y[i] = 1
    return zip(x, y)


def split(data, percent):
    p = percent*len(data)/100
    return (data[:p], data[p:])

data = cargar_datos("iris.data")
b_data = bin_data(data)

redes = [4,5,6,7,8,9,10]

################################################################################

file = open("resutados/e3/resultados.txt")
for r in redes:

file.close()

################################################################################
################################################################################


################################################################################
