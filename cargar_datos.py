def load_data1(filename):
    x = []
    y = []
    with open(filename) as fp:
        for line in fp:
            s = line.split(" ")
            x.append(map(float, s[:-1]))
            y.append(int(s[-1]))

    return x, y

def load_data2(filename):
    # ejercicio 2
