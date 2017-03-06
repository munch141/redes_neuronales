"""
Proyecto 2 de Inteligencia Artificial 2.
Enero-Marzo 2017.
Hecho por:
    Ricardo Munch.       Carnet: 11-10684.
    Valentina Hernandez. Carnet: 10-10352.

Este archivo contiene algunas funciones para graficar los resultados.
"""


import matplotlib.pyplot as plt

def convergencia_por_red(costos, file, show):
    iteraciones = range(len(costos[0]))

    f = plt.figure(1)
    for i, costo in enumerate(costos):
        plt.plot(iteraciones, costo, label=str(i+2))
    plt.title("Curvas de convergencia por numero de neuronas", fontsize = 14,
              color = '0.5', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.xlabel("Iteracion", fontsize = 16, color = '0.50')
    plt.ylabel("Costo", fontsize = 16, color = '0.50')
    plt.legend()
    plt.savefig(file)
    if show:
        plt.show()
    plt.gcf().clear()


def convergencia_por_conjunto(costos, splits, file, show, n):
    iteraciones = range(len(costos[0]))

    f = plt.figure(1)
    for costo, s in zip(costos, splits):
        plt.plot(iteraciones, costo, label=str(s)+"%")
    plt.title("Curvas de convergencia por particion\nRed de {0} neuronas".format(n), fontsize = 14,
              color = '0.5', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.xlabel("Iteracion", fontsize = 16, color = '0.50')
    plt.ylabel("Costo", fontsize = 16, color = '0.50')
    plt.legend()
    plt.savefig(file)
    if show:
        plt.show()
    plt.gcf().clear()


def graficar_circulo(data, archivo, show, n):
    circulo = []
    resto = []
    for p, c in data:
        if c == 1:
            circulo.append(p)
        else:
            resto.append(p)

    if circulo:
        x1, y1 = zip(*circulo)
        plt.plot(x1, y1, 'yo', markersize=1.5, label='circulo')
    if resto:
        x2, y2 = zip(*resto)
        plt.plot(x2, y2, 'co', markersize=1.5, label='rectangulo')
    plt.title("Resultado de la clasificacion - Red de {0} neuronas".format(n), fontsize = 14,
              color = '0.5', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.ylim(0, 20)
    plt.xlim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(archivo)
    if show:
        plt.show()
    plt.gcf().clear()
