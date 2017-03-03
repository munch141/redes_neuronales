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

def grafica(par, clasificacion):
	total = zip(par,clasificacion)
	si = filter(lambda p: p[1] ==0, total)
	si = zip (*si)
	si = list(si[0])
	x1 = zip(*si)
	x1 = list(x1[0])
	y1 = list(x1[1])
	no = filter(lambda q: q[1] ==1, total)
	no = zip(*no)
	no = list(no[0])
	x2 = zip(*no)
	x2 = list(x2[0])
	y2 = list(x2[1])

	f2 = plt.figure(2)
	plt.plot(x1, y1, 'g*', linewidth = 2, label = 'rectangulo')
	plt.hold(True)
	plt.plot(x2,y2, 'ro',linewidth = 2, label = 'circulo')
	plt.ylim(0, 20)
	plt.xlim(0, 20)
	plt.legend()
	plt.title("Validacion de aprendizaje",fontsize = 14
    	, color = '0.75', verticalalignment = 'baseline', horizontalalignment = 'center')
	
	plt.subplots_adjust(0.14)
	plt.savefig("grafica.png")

	plt.show()