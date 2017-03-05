import numpy as np

from graficas import *
from red import Red, Evaluacion


class Results():
    def __init__(self, costs, train_eval, test_eval):
        self.costs = costs
        self.train_eval = train_eval
        self.test_eval = test_eval

    def write_to_file(self, fp):
        fp.write(str(self.train_eval.aciertos))
        fp.write("%, ")
        fp.write(str(self.train_eval.fp))
        fp.write(", ")
        fp.write(str(self.train_eval.fn))
        fp.write(", ")
        fp.write(str(self.test_eval.aciertos))
        fp.write("%, ")
        fp.write(str(self.test_eval.fp))
        fp.write(", ")
        fp.write(str(self.test_eval.fn))
        fp.write("\n")

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


def entrenar(red, train, epocas, mini_batch_size, alpha, \
             test, outfile, outcostos, outtest, outclases):
    costos = red.SGD(train, epocas, mini_batch_size, alpha)
    train_e = red.evaluate(train)
    test_e = red.evaluate(test)
    salida = open(outfile, 'w')
    # porcentaje de aciertos
    salida.write(str(100.0*train_e[0]/len(train)))
    salida.write(", ")
    #falsos positivos
    salida.write(str(train_e[1]))
    salida.write(", ")
    # falsos negativos
    salida.write(str(train_e[2]))
    salida.write(", ")
    salida.write("\n")
    # porcentaje de aciertos
    salida.write(str(100.0*test_e[0]/len(test)))
    salida.write(", ")
    # falsos positivos
    salida.write(str(test_e[1]))
    salida.write(", ")
    # falsos negativos
    salida.write(str(test_e[2]))
    salida.write("\n")
    salida.close()



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

"""train_sets = [c1, c2, c3, c4, c5, c6]
redes = [r2, r3, r4, r5, r6, r7, r8, r9, r10]
resultados = {}
for i, s in enumerate(train_sets):
    file = "resultados/c"+str(i+1)+".csv"
    fp = open(file, 'w')
    for red in redes:
        costos = red.SGD(s, 400, 100, 0.01)
        train_eval = red.evaluate(s)
        test_eval = red.evaluate(ct)
        ident = (red.sizes[1], i)
        resultados[ident] = Results(costos, train_eval, test_eval)
        resultados[ident].write_to_file(fp)


        graficar_circulo(zip(ct, resultados[ident].test_eval.clases))

    fp.close()
"""
costos = r3.SGD(c6, 400, 2000, 0.9)
convergencia(costos)
puntos = zip(*ct)[0]
res = r3.classification(puntos)
res2 = r3.evaluate(c6)
rec = len([1 for (x,y) in c6 if y == 0])
print "rectangulo: ", rec
print "class rec : ", res2
graficar_circulo(zip(puntos, res))

#epocas = 400
#alpha = 0.01




