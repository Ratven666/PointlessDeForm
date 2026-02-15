from app.base.scan.Scan import Scan

import numpy as np

def scan_to_numpy(scan):
    return np.array([[p.x, p.y, p.z] for p in scan])


from sklearn.cluster import DBSCAN

def dbscan_cluster_points(scan, eps=0.01, min_samples=10):
    """
    scan        – ваш объект Scan
    eps         – радиус окрестности (в тех же единицах, что и координаты)
    min_samples – минимальное число точек в окрестности для ядра
    """
    X = scan_to_numpy(scan)  # shape (N, 3)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)  # shape (N,)

    # labels = -1 для шума, 0,1,2,... – кластеры
    return labels


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dbscan_clusters_3d(scan, labels):
    X = scan_to_numpy(scan)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in unique_labels:
        mask = labels == k
        if k == -1:
            ax.scatter(x[mask], y[mask], z[mask], s=1, c='k', alpha=0.1, label='noise')
        else:
            ax.scatter(x[mask], y[mask], z[mask], s=5, color=colors(k), label=f'cluster {k}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best', markerscale=4)
    plt.tight_layout()
    plt.show()

    return fig, ax







if __name__ == "__main__":
    scan = Scan(scan_name="L1")
    scan.import_points_from_file(file_path="src/L1.las")

    print(scan)
    # scan.plot()
    labels = dbscan_cluster_points(scan, eps=0.02, min_samples=5)

    # можно сохранить метки в ScanPoint
    for p, lbl in zip(scan, labels):
        setattr(p, "cluster_id", int(lbl))

    # for p in scan:
    #     print(p, p.cluster_id)

    plot_dbscan_clusters_3d(scan, labels)