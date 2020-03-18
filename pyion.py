

from matplotlib import pyplot
import csv
import numpy as np


def plot_delta(file_name, a, b, block=True):
    data = np.array([])
    with open(file_name) as f:
        data = []
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            # data.append([float(row[0]), float(row[3])])
            data.append(np.array(list(filter(None, row))).astype(np.float))

        fig, ax = pyplot.subplots(2, 1)
        ax[0]: pyplot.Axes
        ax[0].grid(True)
        ax[1].grid(True)

        # ax.set_ylim([-100, 100])
        # t = np.linspace(-1.0, 1.0, len(data))
        # print(t.shape, len(data))
        # p = np.polyfit(t, data, 8)
        # fx = np.polyval(p, t)
        data = np.array(data).T

        # ax[0].plot((data[0] - data[0][0]) * 0.001 / 3600.0, data[b] - data[a], '.')
        # ax[0].plot(data[b] - data[a], '.')
        ax[0].plot((data[b] - data[b, 0]) * 0.001 / 60.0, data[a], '.')
        # ax[1].plot(data[b], data[a], '.')
        ax[0].set_ylim(-10.0, 10.0)
        # ax.set_ylim(-1.0, 1.0)
        pyplot.show(block=block)
    return data


def load_data(file_name):
    data = np.array([])
    with open(file_name) as f:
        data = []
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            data.append(np.array(list(filter(None, row))).astype(np.float))
        data = np.array(data).T
    return data


def plot_polar(data):
    ax = pyplot.subplot(projection='polar')
    ax: pyplot.PolarAxes
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    theta = data[7]
    r = 0.5 * np.pi - data[8]
    # color = (data[5] - np.min(data[5])) / np.max(data[5])
    # color = cm.hot(color)

    ax.plot(theta, r, '.')
    # for i in range(len(theta) // 10):
    #     ax.plot(theta[i], r[i], '.', color=color[i])
    ax.grid(True)
    pyplot.show()
    # ax.polar

def polar2rect(r, phase):
    return r * np.exp(1.0j * phase)


data = load_data('/home/taran/work/pyion/delta1.txt')
# plot_polar(data)
traces = polar2rect(0.5 * np.pi - data[8], data[7])

n = 1024
points_num = data[3].shape[0]

x = ((-np.real(traces) / np.pi + 0.5) * n).astype(int)
y = ((np.imag(traces) / np.pi + 0.5) * n).astype(int)
z = data[4]
# z = data[3] - np.min(data[3])
# z /= np.std(z)
# z = np.log(z)
pic = np.zeros((n, n))
counters = np.zeros((n, n)) + 1.0

threshold = np.std(z)
z = np.clip(z, -threshold, threshold)

for i in range(points_num):
    pic[x[i], y[i]] += z[i]
    counters[x[i], y[i]] += 1.0
pic /= counters
pic -= np.min(pic)

# pic = np.log(pic)
# pic += np.std(z) * 3
fig, ax = pyplot.subplots(1)
ax: pyplot.Axes
fig: pyplot.Figure

# ax.grid(True)
# ax.plot(x, y, '.')
ax.imshow(pic, cmap='gray')
fig.set_size_inches(16, 16)
pyplot.show()
