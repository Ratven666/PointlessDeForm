import numpy as np
from sklearn.cluster import DBSCAN

class ScanDBSCANClusterizator:

    def __init__(self, scan):
        self.scan = scan

    def __scan_to_numpy(self):
        return np.array([[p.x, p.y, p.z] for p in self.scan])

    def compute_clusters(self, *args, eps=0.01, min_samples=10, **kwargs):
        labels = self.__dbscan_cluster_points(eps=eps, min_samples=min_samples)
        for p, n in zip(self.scan, labels):
            setattr(p, "labels", n)
        return labels


    def __dbscan_cluster_points(self, eps=0.01, min_samples=10):
        """
        eps         – радиус окрестности (в тех же единицах, что и координаты)
        min_samples – минимальное число точек в окрестности для ядра
        """
        X = self.__scan_to_numpy()  # shape (N, 3)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)  # shape (N,)

        # labels = -1 для шума, 0,1,2,... – кластеры
        return labels

if __name__ == "__main__":
    from app.base.scan.Scan import Scan

    scan = Scan("l1")

    scan.import_points_from_file(file_path=r"../../../../src/L1.las")

    s_dbscan_c = ScanDBSCANClusterizator(scan)
    s_dbscan_c.compute_clusters()

    for point in scan:
        print(point, point.labels)