import matplotlib.pyplot as plt

def convergencia(costos, file):
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
    plt.show()


def graficar_circulo(data, archivo):
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
    plt.ylim(0, 20)
    plt.xlim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(archivo)
    plt.show()
