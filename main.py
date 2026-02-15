from app.base.scan.Scan import Scan




############

import numpy as np
from sklearn.neighbors import NearestNeighbors

def scan_to_numpy(scan):
    """Преобразуем Scan в массив N x 3."""
    pts = np.array([[p.x, p.y, p.z] for p in scan])
    return pts

def compute_normals(points_xyz, k=20):
    """
    points_xyz: np.ndarray (N, 3)
    k: число соседей для оценки нормали
    return: np.ndarray (N, 3) нормали (единичные векторы)
    """
    N = points_xyz.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)

    # kNN по евклиду
    nn = NearestNeighbors(n_neighbors=min(k, N), algorithm='kd_tree')
    nn.fit(points_xyz)
    distances, indices = nn.kneighbors(points_xyz)

    for i in range(N):
        neigh_idx = indices[i]
        neigh_pts = points_xyz[neigh_idx]

        # центрируем
        centroid = neigh_pts.mean(axis=0)
        centered = neigh_pts - centroid

        # PCA через SVD: наименьшее собственное значение даёт нормаль
        # (u, s, vh) = SVD(centered), нормаль = последний вектор vh
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1, :]

        # нормализуем
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n = n / n_norm

        normals[i] = n

    return normals

def compute_scan_normals(scan, k=20):
    """
    Возвращает dict: point -> normal (np.array shape (3,))
    или список нормалей в том же порядке, что и scan.
    """
    pts = scan_to_numpy(scan)
    normals = compute_normals(pts, k=k)

    # если хотите положить нормаль внутрь ScanPoint
    for p, n in zip(scan, normals):
        # добавьте в ScanPoint атрибут, если его нет
        setattr(p, "normal", n)

    return normals


if __name__ == "__main__":
    scan = Scan(scan_name="L1")
    scan.import_points_from_file(file_path="src/L1.las")

    print(scan)
    # scan.plot()

    normals = compute_scan_normals(scan, k=30)

    print(normals)
