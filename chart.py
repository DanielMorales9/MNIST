import os
import matplotlib.pylab as plt
import numpy as np


def create_charts(label, pixels, file_name):
    fig = plt.figure(num=None, figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.title("label {0}".format(label))
    pix = np.copy(pixels)
    pix = pix.reshape((28, 28))
    mat = np.matrix(pix)
    plt.imshow(mat, cmap='Greys_r')
    plt.colorbar()
    plt.savefig(os.path.join("charts", str(file_name)))
