from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def visualize_mpm(mpm):
    X, Y = np.meshgrid(np.arange(0, int(mpm.shape[1]/10)), np.arange(0, int(mpm.shape[0]/10)))

    fig, ax = plt.subplots()

    U = mpm[::10,::10,1]
    V = mpm[::10,::10,0]
    q = ax.quiver(X, Y, U, V, scale=50.0)


    plt.show()
    pass

magnitude = np.zeros((256, 256, 3))
magnitude[128, 128] = 255
magnitude = gaussian_filter(magnitude, sigma=6, cval=True)

mpm = np.load('../../sample_mpm.npy')
visualize_mpm(mpm)