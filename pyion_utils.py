
from matplotlib import pyplot
import numpy as np


def pyion_load(file_name):
    data: np.array
    data = np.loadtxt(file_name, delimiter='\t')
    print(data.shape)
    ion = data[:, 3]
    azm = data[:, 7]
    elv = data[:, 8]
    data = {'ion': ion, 'azm': azm, 'elv': elv}
    return data

def polar2rect(r, phase):
    return r * np.exp(1.0j * phase)

def pyion_plotmap(data: dict, n: int = 128):
    ion = data['ion']
    azm = data['azm']
    elv = data['elv']
    points_num = ion.shape[0]
    
    points = polar2rect(0.5 * np.pi - elv, azm)
    x = ((-np.real(points) / np.pi + 0.5) * n).astype(int)
    y = ((np.imag(points) / np.pi + 0.5) * n).astype(int)

    picture = np.zeros((n, n))
    counters = np.zeros((n, n)) + 1.0

    threshold = np.std(ion)
    ion = np.clip(ion, -threshold, threshold)

    for i in range(points_num):
        picture[x[i], y[i]] += ion[i]
        counters[x[i], y[i]] += 1.0
    picture /= counters
    picture -= np.min(picture)

    fig, ax = pyplot.subplots(1)
    ax: pyplot.Axes
    fig: pyplot.Figure
    ax.imshow(picture, cmap='gray')
    fig.set_size_inches(16, 16)
    pyplot.show()
