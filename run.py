import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.3 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def plot(x):
    d = center(x)
    methods = ['sklearn', 'eig', 'eigh']
    ncols = len(methods)
    fig, axs = plt.subplots(ncols=ncols, subplot_kw=dict(projection='3d'), figsize=(6*ncols, 6))

    for i, method in enumerate(methods):
        if method == 'sklearn':
            pc1, pc2, pc3 = get_pcs_sk(d)
            title = 'sklearn.decomposition.PCA().fit(x.T)'
        elif method == 'eig':
            pc1, pc2, pc3 = get_pcs_eig(d)
            title = 'la.eig(np.cov(x))'
        elif method == 'eigh':
            pc1, pc2, pc3 = get_pcs_eigh(d)
            title = 'la.eigh(np.cov(x))'

        ax = axs[i]
        ax.scatter(d[0, :], d[1, :], d[2, :], marker='o', zorder=0)
        for j, pc in enumerate([pc1, pc2, pc3]):
            scale = 6.5
            pc_x = pc[0]*scale
            pc_y = pc[1]*scale
            pc_z = pc[2]*scale
            ax.quiver(0, 0, 0, pc_x, pc_y, pc_z, zorder=40, color='k')
            offset = 0.1
            ax.text(pc_x + offset, pc_y + offset, pc_z + offset, f'PC{j + 1}', fontsize=16, zorder=60)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        ax.view_init(20, 250 + 5)
        ax.set_box_aspect([1, 1, 1])
        set_axes_equal(ax)
    plt.savefig('pca.pdf', bbox_inches='tight')
    plt.show()

def gen_data():
    np.random.seed(0)
    mu = np.array([10, 13, 7])
    sigma = np.array([
        [5, -1, 2],
        [-1, 3, -1],
        [2, -1, 1]
        ])
    x = np.random.multivariate_normal(mu, sigma, size=1000)
    return x.T

def center(x):
    mu = x.mean(axis=1, keepdims=True)
    return x - mu

def get_pcs_eigh(x):
    lamda, V = la.eigh(np.cov(x))
    pc1 = V[:, -1] / la.norm(V[:, -1])
    pc2 = V[:, -2] / la.norm(V[:, -2])
    pc3 = V[:, -3] / la.norm(V[:, -3])
    return pc1, pc2, pc3

def get_pcs_eig(x):
    lamda, V = la.eig(np.cov(x))
    pc1 = V[:, 0] / la.norm(V[:, 0])
    pc2 = V[:, 1] / la.norm(V[:, 1])
    pc3 = V[:, 2] / la.norm(V[:, 2])
    return pc1, pc2, pc3

def get_pcs_sk(x):
    pca = PCA().fit(x.T)
    V = pca.components_
    lamda = pca.singular_values_
    pc1 = V[0, :] / la.norm(V[0, :])
    pc2 = V[1, :] / la.norm(V[1, :])
    pc3 = V[2, :] / la.norm(V[2, :])
    return pc1, pc2, pc3

if __name__ == '__main__':
    plot(gen_data())
