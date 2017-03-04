import matplotlib.pyplot as plt

def convergencia(costos):
    iteraciones = [i for i in range(len(costos))]

    f1 = plt.figure(1)
    plt.plot(iteraciones, costos, 'gx')
    plt.title("Curva de convergencia de la Red Neural", fontsize = 14
    , color = '0.75', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.xlabel("Iteracion", fontsize = 16, color = '0.50')
    plt.ylabel("Costo", fontsize = 16, color = '0.50')
    plt.subplots_adjust(0.14)
    plt.savefig("convergencia.png")

def graficar_circulo(data, archivo="plots/grafica.png"):
    circulo = []
    resto = []
    for p, c in data:
        if c == 1:
            circulo.append(p)
        else:
            resto.append(p)

    x1, y1 = zip(*circulo)
    x1, y1 = list(x1), list(y1)
    x2, y2 = zip(*resto)
    x2, y2 = list(x2), list(y2)

    color1 = "33ff8d"
    color2 = "33fffc"

    plt.plot(x1, y1, 'yo', markersize=1.5, label='circulo')
    plt.plot(x2, y2, 'co', markersize=1.5, label='rectangulo')
    plt.ylim(0, 20)
    plt.xlim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(archivo)

    plt.show()