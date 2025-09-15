# % Script for Project 1 part 3
# % ECE 5370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import PCA

# loading the dataset
f = open('EEG Eye State.json','rt')
dataset = json.load(f)
f.close()
data = np.array(dataset['data'])

# filter outliers
rng=1.5
for i in range(0,np.shape(data)[1]):
    sts = np.quantile(data[:,i],[.01,.5,.99])
    arr = data[:,i] > sts[1] + rng*(sts[2]-sts[1])
    data[arr,i] = sts[1] + rng*(sts[2]-sts[1])
    arr = data[:, i] < sts[1] + rng * (sts[0] - sts[1])
    data[arr, i] = sts[1] + rng * (sts[0] - sts[1])

# trim to 2000 timepoints
data = data[0:1999,:]

# initialize plots
f, ax = plt.subplots(2,2)
fnum = plt.get_fignums()[-1]

#define a class that will update the base plots and in which you will define your mouse button callback code
class Project1:
    def __init__(self, data, ax):
        self.data = data
        self.ax = ax
        plt.ion()

    def UpdatePlot(self):
        plt.axes(ax[0][0])
        plt.cla()
        plt.plot(data[:, 0])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 0]) - np.min(data[:, 0])) + np.min(data[:, 0]), '--')
        plt.title('Sensor 1')
        plt.axes(ax[0][1])
        plt.cla()
        plt.plot(data[:, 1])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 1]) - np.min(data[:, 1])) + np.min(data[:, 1]), '--')
        plt.title('Sensor 2')
        plt.axes(ax[1][0])
        plt.cla()
        plt.plot(data[:, 2])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 2]) - np.min(data[:, 2])) + np.min(data[:, 2]), '--')
        plt.title('Sensor 3')
        plt.axes(ax[1][1])
        plt.cla()
        h1 = ax[1][1].plot(data[:, 0], label='Sensor_1')
        h2 = ax[1][1].plot(data[:, 1], label='Sensor_2')
        h3 = ax[1][1].plot(data[:, 2], label='Sensor_3')
        ax[1][1].legend()
    #def on_mouse_click(self, event):

if __name__=='__main__':
    # initiate a Project1 object
    # with (open(r'C:\Users\Victor\PycharmProjects\ECE_4370\Project1\DimensionalityReduction.json', 'rt') as f:
    #   D = json.load(f)
    #   f.close()
    # fig, ax = plot.subplots()
    # pca_data = PCA(D)

    p = Project1(data, ax)
    p.UpdatePlot()

    # Need an infinite while-loop to use the figure interactively
    plt.show()
    while 1:
        if plt.fignum_exists(fnum) == False:
            break
        f.canvas.draw_idle()
        f.canvas.start_event_loop(0.3)
