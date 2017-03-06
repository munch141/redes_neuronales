import numpy as np

from graficas import *
from red import Red


def cargar_datos(filename):
    x = []
    y = []
    with open(filename) as fp:
        for line in fp:
            s = line.split(" ")
            x.append(np.array(map(float, s[:-1])).reshape(2,1))
            y.append(int(s[-1]))

    return zip(x, y)


def generar_conjunto_entrenamiento(n):
    dentro = n/2
    fuera = n - dentro
    puntos = []
    clases = []
    np.random.seed(3)
    while (dentro != 0 or fuera != 0):
        p = np.random.uniform(0, 20, 2).reshape(2,1)
        c = (p[0][0]-10)*(p[0][0]-10) + (p[1][0]-10)*(p[1][0]-10)
        if (c <= 36 and dentro > 0):
            puntos.append(p);
            clases.append(1);
            dentro -= 1
        elif (fuera > 0):
            puntos.append(p);
            clases.append(0);
            fuera -= 1
    return zip(puntos, clases)


def generar_conjunto_prueba():
    r = np.arange(0,20,0.2)
    puntos = [(x,y) for x in r for y in r]
    clases = []
    for p in puntos:
        c = (p[0]-10)*(p[0]-10) + (p[1]-10)*(p[1]-10)
        if c <= 36:
            clases.append(1);
        else:
            clases.append(0);
    return zip(puntos, clases)


def falsos_positivos(a, y):
    return sum(1 for (i, j) in zip(a, y) if i == 1 and j != 1)

def falsos_negativos(a, y):
    return sum(1 for (i, j) in zip(a, y) if i == 0 and j != 0)

################################################################################
################################################################################

c1 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N500.txt")
c2 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N1000.txt")
c3 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N2000.txt")
c4 = generar_conjunto_entrenamiento(500)
c5 = generar_conjunto_entrenamiento(1000)
c6 = generar_conjunto_entrenamiento(2000)

r2 = Red([2, 2, 2])
r3 = Red([2, 3, 2])
r4 = Red([2, 4, 2])
r5 = Red([2, 5, 2])
r6 = Red([2, 6, 2])
r7 = Red([2, 7, 2])
r8 = Red([2, 8, 2])
r9 = Red([2, 9, 2])
r10 = Red([2, 10, 2])

ct = generar_conjunto_prueba()

train_sets = [c1, c2, c3, c4, c5, c6]
redes = [r2, r3, r4, r5, r6, r7, r8, r9, r10]

aciertos = []
costos = [[],[],[],[],[],[],[],[],[]]
for i, train_data in enumerate(train_sets):
    file = open("resultados/c"+str(i+1)+".csv", 'w')
    total_aciertos = 0
    for j, r in enumerate(redes):
        costo = r.SGD(train_data, 400, len(train_data)/4, 2.0)
        costos[j].append(costo)
        train_eval = r.accuracy(train_data)
        puntos = [np.array(p).reshape(2,1) for p in zip(*train_data)[0]]
        res = r.classify(puntos)
        file.write(str(train_eval))
        file.write(",")
        file.write(str(falsos_positivos(res, zip(*train_data)[1])))
        file.write(",")
        file.write(str(falsos_negativos(res, zip(*train_data)[1])))
        file.write(",")        

        test_eval = r.accuracy(ct)
        puntos = [np.array(p).reshape(2,1) for p in zip(*ct)[0]]
        res = r.classify(puntos)
        file.write(str(test_eval))
        file.write(",")
        file.write(str(falsos_positivos(res, zip(*ct)[1])))
        file.write(",")
        file.write(str(falsos_negativos(res, zip(*ct)[1])))
        file.write("\n")

        total_aciertos += test_eval
    aciertos.append(total_aciertos)
    file.close()

index = aciertos.index(max(aciertos))
best_train_data = train_sets[index]
for i in range(len(costos)):
    convergencia(costos[i][index])

#graficar_circulo(zip(puntos, res))
"""
costos = r7.SGD(c6, 1000, len(c6)/4, 2.0)

puntos = [np.array(p).reshape(2,1) for p in zip(*ct)[0]]
res = r7.classify(puntos)
graficar_circulo(zip(puntos, res))

puntos = [np.array(p).reshape(2,1) for p in zip(*c6)[0]]
res = r7.classify(puntos)
graficar_circulo(zip(puntos, res))
"""




