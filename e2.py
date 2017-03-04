import numpy as np

from graficas import grafica

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
    return puntos, clases

x, y = generar_conjunto(2000)
grafica(x, y)