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


data = cargar_datos("iris.data")
b_data = bin_data(data)

binarios = []
clasificadores = []
for i in range(4,11):
    binarios.append(Red([4, i, 2]))
    clasificadores.append(Red([4, i, 3]))



