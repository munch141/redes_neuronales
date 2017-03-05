import matplotlib.pyplot as plt

def convergencia(costos):
    iteraciones = [i for i in range(len(costos))]

    f1 = plt.figure(1)
    plt.plot(iteraciones, costos)
    plt.title("Curva de convergencia de la Red Neural", fontsize = 14
    , color = '0.75', verticalalignment = 'baseline', horizontalalignment = 'center')
    plt.xlabel("Iteracion", fontsize = 16, color = '0.50')
    plt.ylabel("Costo", fontsize = 16, color = '0.50')
    plt.subplots_adjust(0.14)
    #plt.savefig("convergencia.png")
    plt.show()


def graficar_circulo(data, archivo="plots/grafica.png"):
    circulo = []
    resto = []
    for p, c in data:
        if c == 1:
            circulo.append(p)
        else:
            resto.append(p)


    if circulo:
        x1, y1 = zip(*circulo)
        x1 = [list(x)[0] for x in x1]
        y1 = [list(y)[0] for y in y1]
        plt.plot(x1, y1, 'yo', markersize=1.5, label='circulo')  
    if resto:
        x2, y2 = zip(*resto)
        x2 = [list(x)[0] for x in x2]
        y2 = [list(y)[0] for y in y2]
        plt.plot(x2, y2, 'co', markersize=1.5, label='rectangulo')
    plt.ylim(0, 20)
    plt.xlim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(archivo)
    plt.show()
