import numpy as np

from graficas import graficar_circulo
from red import Red

def cargar_datos(filename):
    x = []
    y = []
    with open(filename) as fp:
        for line in fp:
            s = line.split(" ")
            x.append(map(float, s[:-1]))
            y.append(int(s[-1]))

    return zip(x, y)


def generar_conjunto(n):
    dentro = n/2
    fuera = n - dentro
    puntos = []
    clases = []
    while (dentro != 0 or fuera != 0):
        p = np.random.uniform(0, 20, 2)
        c = (p[0]-10)*(p[0]-10) + (p[1]-10)*(p[1]-10)
        if (c <= 36 and dentro > 0):
            puntos.append(p);
            clases.append(1);
            dentro -= 1
        elif (fuera > 0):
            puntos.append(p);
            clases.append(0);
            fuera -= 1
    return zip(puntos, clases)


c1 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N500")
c2 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N1000")
c3 = cargar_datos("datosP2EM2017/datos_P2_EM2017_N2000")
c4 = generar_conjunto(500)
c5 = generar_conjunto(1000)
c6 = generar_conjunto(2000)

r2 = Red([2, 2, 2])
r3 = Red([2, 3, 2])
r4 = Red([2, 4, 2])
r5 = Red([2, 5, 2])
r6 = Red([2, 6, 2])
r7 = Red([2, 7, 2])
r8 = Red([2, 8, 2])
r9 = Red([2, 9, 2])
r10 = Red([2, 10, 2])

r2.SGD()
