from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import cv2

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def getPeaks_getIndicatedPoints(mpm, mag_th=0.3):
    mag = np.sqrt(np.sum(np.square(mpm), axis=-1))
    map_left = np.zeros(mag.shape)
    map_right = np.zeros(mag.shape)
    map_top = np.zeros(mag.shape)
    map_bottom = np.zeros(mag.shape)
    map_left_top = np.zeros(mag.shape)
    map_right_top = np.zeros(mag.shape)
    map_left_bottom = np.zeros(mag.shape)
    map_right_bottom = np.zeros(mag.shape)
    map_left[1:, :] = mag[:-1, :]
    map_right[:-1, :] = mag[1:, :]
    map_top[:, 1:] = mag[:, :-1]
    map_bottom[:, :-1] = mag[:, 1:]
    map_left_top[1:, 1:] = mag[:-1, :-1]
    map_right_top[:-1, 1:] = mag[1:, :-1]
    map_left_bottom[1:, :-1] = mag[:-1, 1:]
    map_right_bottom[:-1, :-1] = mag[1:, 1:]
    peaks_binary = np.logical_and.reduce((
        mag >= map_left,
        mag >= map_right,
        mag >= map_top,
        mag >= map_bottom,
        mag >= map_left_top,
        mag >= map_left_bottom,
        mag >= map_right_top,
        mag >= map_right_bottom,
        mag > mag_th
    ))
    _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
    center = center[1:]
    result = []
    for center_cell in center.astype('int'):
        vec = mpm[center_cell[1], center_cell[0]]
        mag_value = mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        # print(vec)
        x = 0 if vec[1] == 0 else 5 * (vec[1] / vec[2])
        y = 0 if vec[0] == 0 else 5 * (vec[0] / vec[2])
        x = int(x)
        y = int(y)
        result.append([center_cell[0], center_cell[1], center_cell[0] + x, center_cell[1] + y, mag_value])
    return np.array(result)


def visualize3D(pre, cur, mpm, z_value=5, l_th=0.4):
    mpm = mpm.astype('float32')
    if mpm.shape[0] == 3:
        mpm = mpm.transpose(1, 2, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                    bottom=False,
                    left=False,
                    right=False,
                    top=False)

    z_range = (pre.shape[1] + pre.shape[0]) // 2
    x = np.arange(0, pre.shape[1])
    y = np.arange(0, pre.shape[0])
    z = np.arange(0, z_range)
    ax.set_xlim(0, len(x))
    ax.set_ylim(0, len(y))
    ax.set_zlim(0, len(z))
    X, Y, Z = np.meshgrid(x, y, z)
    X2, Y2 = np.meshgrid(x, y)
    ax.contourf(X2, Y2, pre, 255, cmap='binary_r', offset=0)
    ax.contourf(X2, Y2, cur, 255, cmap='binary_r', offset=z[-1], alpha=0.1)

    result = getPeaks_getIndicatedPoints(mpm)
    for d in result:
        # ax.plot([d[0], d[2]], [d[1], d[3]], [0, z[-1]])
        a = Arrow3D([d[0], d[2]], [d[1], d[3]], [z[-1], 0], mutation_scale=5,
                    lw=5, arrowstyle="-|>,head_length=1,head_width=0.7", color=np.random.rand(3,), zorder=256)
        ax.add_artist(a)

    plt.show()

    return


pre = cv2.imread('/home/junya/MPM/MPM/data/train_imgs/F18/0091.png', -1)#[500:600, 500:600]
cur = cv2.imread('/home/junya/MPM/MPM/data/train_imgs/F18/0100.png', -1)#[500:600, 500:600]
mpm = np.load('/home/junya/MPM/MPM/data/train_mpms/F18/009/0091.npy')#[500:600, 500:600]
vismpm = visualize3D(pre, cur, mpm)

# def visualize_mpm(mpm, magnitude):
#     X, Y = np.meshgrid(np.arange(0, int(mpm.shape[1])), np.arange(0, int(mpm.shape[0])))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#
#     U = mpm[:, :, 1]
#     V = mpm[:, :, 0]
#
#     # for x in range(mpm.shape[1]):
#     #     for y in range(mpm.shape[0]):
#     plt.quiver(X, Y, U, V, magnitude, cmap='winter')
#     plt.colorbar(cmap='winter')
#     plt.show()
#     pass

# magnitude = np.zeros((64, 64))
# mpm = np.zeros((64, 64, 3))
# magnitude[32, 32] = 255
# magnitude = gaussian_filter(magnitude, sigma=6)
# magnitude = magnitude/magnitude.max()
#
# for x in range(64):
#     for y in range(64):
#         vec = np.array([42, 42]) - np.array([y, x])
#         vec = np.append(vec, 5)
#         vec = vec / np.linalg.norm(vec) * magnitude[y, x] * 2
#         mpm[y, x] = vec
#
# visualize_mpm(mpm[15:-15, 15:-15], magnitude[15:-15, 15:-15])
