import random
from collections import namedtuple

import numpy as np


class Red(object):
    def __init__(self, sizes):
        """
        num_layers : cantidad de capas de red.
        sizes      : arreglo con las cantidades de neuronas por capa.
        biases     : arreglo de arreglos con los bias de cada capa.
        weights    : arreglo de matrices que contiene, para cada capa, los pesos
                     de cada conexion en la red. Para una capa intermedia
                     'i', la posicion (x,y) representa el peso del enlace entre
                     la neurona 'x' de la capa i y la neurona 'y' de la capa i-1.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Devuelve la salida de la red para un vector de entrada 'a'.
        
        a : vector con los valores de entrada.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def cost(self, y, a):
        """
        Costo de una instancia.
        """
        return np.linalg.norm(vectorized_result(y)-a)**2

    def total_cost(self, data):
        """
        Costo total sobre data.
        """
        s = sum([self.cost(y, self.feedforward(x))
                 for x, y in data])
        return s / (2*len(data))

    def SGD(self, training_data, epochs, mini_batch_size, alpha):
        """
        Descenso del gradiente estocastico. Se divide el conjunto de
        entrenamiento en lotes aleatorios y se aplica descenso del gradiente con
        cada lote. Termina cuando se hayan terminado todas las epocas. Una epoca
        termina cuando se aplica el descenso del gradiente estocastico a todos
        los lotes del conjunto de entrenamiento.
        
        training_data : arreglo de tuplas de forma (x,y) donde 'x' es un vector
                        con los valores de entrada y 'y' es el indice de la
                        neurona con el mayor valor de activacion.
        epochs : cantidad de epocas para entrenar.
        mini_batch_size : tamano de los lotes a tomar del conjunto de
                          entrenamiento.
        alpha : taza de aprendizaje.

        """
        n = len(training_data)
        costos = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            costos.append(self.total_cost(training_data))
            print "epoca {0} completada".format(j)
        return costos

    def update_mini_batch(self, mini_batch, alpha):
        """
        Actualiza los pesos y las tendencias (bias) de la red para aplicando
        backpropagation a los ejemplos de un lote. Para cada ejemplo se aplica
        backpropagation para obtener los vectores de error y activacion de cada
        capa y los resultados se van sumando en 'nabla_w' y 'nabla_b'. Despues
        de calcular esto, se actualizan los vectores de peso y bias.

        mini_batch : conjunto de ejemplos para calcular las actualizaciones.
        alpha : tasa de aprendizaje.
        """
        # vector para guardar la sumatoria del producto de los errores 'delta' y
        # las activaciones 'a' para cada capa
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # vector para guardar la sumatoria de los errores 'delta' en cada capa
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(alpha/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(alpha/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """
        Implementacion de backpropagation.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        #nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_w[-1] = delta * activations[-2].transpose()

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = delta * activations[-l-1].transpose()
        return (nabla_b, nabla_w)

    def classify(self, data):
        return [np.argmax(self.feedforward(x)) for x in data]

    def accuracy(self, test_data):
        puntos = [np.array(p).reshape(2,1) for p in zip(*test_data)[0]]
        a = self.classify(puntos)
        y = zip(*test_data)[1]
        return 100.0*sum(1 for (x, y) in zip(a, y) if x == y)/len(puntos)

    def cost_derivative(self, output_activations, y):
        return (output_activations-vectorized_result(y))


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e
