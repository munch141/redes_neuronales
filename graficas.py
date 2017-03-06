import matplotlib.pyplot as plt

def convergencia(costos, file):
    iteraciones = [i for i in range(len(costos))]

    f = plt.figure(1)
    plt.plot(iteraciones, costos)
    plt.title("Curvas de convergencia de la funcion de costo", fontsize = 14,
              color = '0.90', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.xlabel("Iteracion", fontsize = 16, color = '0.50')
    plt.ylabel("Costo", fontsize = 16, color = '0.50')
    plt.subplots_adjust(0.14)
    plt.saveplot(file)


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
    plt.saveplot(archivo)
