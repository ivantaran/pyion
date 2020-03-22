
from matplotlib import pyplot
import numpy as np


def pyion_load(file_name):
    data: np.array
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.float32)
    print(data.shape)
    ion = data[:, 3]
    azm = data[:, 7]
    elv = data[:, 8]
    mx = np.concatenate((azm, elv))
    mx = np.reshape(mx, (2, -1))
    my = np.reshape(ion, (1, -1))
    assert mx.shape[0] == 2
    assert my.shape[0] == 1
    return mx, my


def polar2rect(r, phase):
    return r * np.exp(1.0j * phase)


def pyion_plotmap(data: dict, n: int = 128, cmap='bone'):
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
    ax.imshow(picture, cmap=cmap)
    fig.set_size_inches(16, 16)
    pyplot.show()


def pyion_debug_samples(n=1000, azm_range=None, freq=20.0, dtype=np.float32):
    ny = n
    nx = n * 2

    elv = np.linspace(0.0, np.pi * 0.5, n, dtype=dtype).reshape((n, 1))
    if azm_range is None:
        azm = np.zeros((n, 1), dtype=dtype)
    else:
        left, right = azm_range
        azm = np.linspace(2.0 * np.pi * left, 2.0 * np.pi * right, n, dtype=dtype).reshape((n, 1))

    mx = np.concatenate((azm, elv))

    t = np.linspace(0.0, 1.0, ny, dtype=dtype)
    amp = t * t + 0.5
    ion = np.sin(2.0 * np.pi * t * freq) * amp
    my = ion.reshape((ny, 1))

    assert mx.shape == (nx, 1)
    assert my.shape == (ny, 1)

    return mx, my


def pyion_debug_samples_m(m=1000, azm_range=None, freq=20.0, dtype=np.float32):
    elv = np.linspace(0.0, np.pi * 0.5, m, dtype=dtype).reshape((1, m))
    if azm_range is None:
        azm = np.zeros((1, m), dtype=dtype)
    else:
        left, right = azm_range
        azm = np.linspace(2.0 * np.pi * left, 2.0 * np.pi * right, m, dtype=dtype).reshape((1, m))

    mx = np.concatenate((azm, elv))

    t = np.linspace(0.0, 1.0, m, dtype=dtype)
    amp = t * t + 0.5
    ion = np.sin(2.0 * np.pi * t * freq) * amp
    my = ion.reshape((1, m))

    assert mx.shape == (2, m)
    assert my.shape == (1, m)

    return mx, my


def pyion_debug_samples2(n=1000, dtype=np.float32):
    mx1, my1 = pyion_debug_samples(n, azm_range=(-0.2, 0.2), freq=20, dtype=dtype)
    mx2, my2 = pyion_debug_samples(n, azm_range=(0.4, 0.6), freq=10, dtype=dtype)
    mx = np.concatenate((mx1, mx2))
    my = np.concatenate((my1, my2))

    assert mx.shape == (n * 4, 1)
    assert my.shape == (n * 2, 1)

    return mx, my


def pyion_debug_samples2_m(m=1000, dtype=np.float32):
    mx1, my1 = pyion_debug_samples_m(m, azm_range=(-0.2, 0.2), freq=20, dtype=dtype)
    mx2, my2 = pyion_debug_samples_m(m, azm_range=(0.4, 0.6), freq=10, dtype=dtype)
    mx = np.concatenate((mx1, mx2), axis=1)
    my = np.concatenate((my1, my2), axis=1)

    assert mx.shape == (2, m * 2)
    assert my.shape == (1, m * 2)

    return mx, my


def pyion_shuffle_samples(mx, my):
    m = mx.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_mx = mx[:, permutation]
    shuffled_my = my[:, permutation]
    return shuffled_mx, shuffled_my


def pyion_load_parameters(file_name):
    parameters = np.load(file_name)
    return parameters


def pyion_square_to_azmelv(n=128, dtype=np.float32):
    t = np.arange(0, n, dtype=dtype)
    x, y = np.meshgrid(t, t)
    x = np.ravel(x) - n * 0.5
    y = np.ravel(y) - n * 0.5
    s = np.stack((x, y))
    elv = np.linalg.norm(s, axis=0)
    elv = (1.0 - elv / np.max(elv)) * np.pi * 0.5
    azm = np.arctan2(x, y)
    mx = np.reshape((azm, elv), (n * n * 2, 1))
    return mx


def pyion_azmelv_mesh(n_azm=128, n_elv=64, dtype=np.float32):
    azm = np.linspace(0.0, 2.0 * np.pi, n_azm, dtype=dtype)
    elv = np.linspace(np.pi * 0.07, 0.5 * np.pi, n_elv, dtype=dtype)
    azm, elv = np.meshgrid(azm, elv)
    azm = np.reshape(azm, (1, -1))
    elv = np.reshape(elv, (1, -1))
    mx = np.concatenate((azm, elv))
    return mx
