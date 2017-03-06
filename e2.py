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
redes = [2, 3, 4, 5, 6, 7, 8, 9, 10]

costos = [[],[],[],[],[],[],[],[],[]]
clasificaciones = [[],[],[],[],[],[],[],[],[]]
for i, train_data in enumerate(train_sets):
    file = open("resultados/e2/c"+str(i+1)+".csv", 'w')
    for j, r in enumerate(redes):
        print "entrenando red de {0} neuronas:\n".format(r)
        red = Red([2,r,2])
        costo = red.SGD(train_data, 400, len(train_data)/4, 0.1)
        costos[j].append(costo)
        train_eval = red.accuracy(train_data, 2)
        puntos = [np.array(p).reshape(2,1) for p in zip(*train_data)[0]]
        res = red.classify(puntos)
        file.write(str(train_eval))
        file.write("%,")
        file.write(str(falsos_positivos(res, zip(*train_data)[1])))
        file.write(",")
        file.write(str(falsos_negativos(res, zip(*train_data)[1])))
        file.write(",")        

        test_eval = red.accuracy(ct, 2)
        puntos = [np.array(p).reshape(2,1) for p in zip(*ct)[0]]
        res = red.classify(puntos)
        clasificaciones[j].append(res)
        file.write(str(test_eval))
        file.write("%,")
        file.write(str(falsos_positivos(res, zip(*ct)[1])))
        file.write(",")
        file.write(str(falsos_negativos(res, zip(*ct)[1])))
        file.write("\n")
        print "\n"
    file.close()
    print "\n\nconjunto {0} listo!\n\n".format(i+1)

print "Listo!\n"
index = 3  # despues de realizar varios experimentos, concluimos que
           # el mejor conjunto de entrenamiento es el de 500 datos
           # balanceados porque, estan balanceados (!) y presentaron
           # los mejores resultados en el conjunto de pruebas
best_train_data = train_sets[index]
cs = []
for c in costos:
    cs.append(c[index])
convergencia_por_red(cs, "plots/e2/convergencias.png", False)

puntos = [np.array(p).reshape(2,1) for p in zip(*ct)[0]]
for i, c in enumerate(clasificaciones):
    graficar_circulo(zip(puntos, c[index]), "plots/e2/red"+str(i+2)+".png", False, i+2)




