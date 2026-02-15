from app.base.scan.filters.ScanFilterABC import ScanFilterABC
from app.base.scan.utils.ScanDBSCANClusterizator import ScanDBSCANClusterizator


class ScanFilterByDBSCAN(ScanFilterABC):

    def __init__(self, eps=0.05, min_samples=100, min_cluster_size=None):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

    def __compute_clusters(self, scan):
        s_dbscan_c = ScanDBSCANClusterizator(scan)
        s_dbscan_c.compute_clusters(eps=self.eps, min_samples=self.min_samples)

    def filter(self, scan):
        self.__compute_clusters(scan)
        point_lst = []
        if self.min_cluster_size is None:
            for point in scan:
                if point.labels != -1:
                    point_lst.append(point)
        else:
            points = {}

        return point_lst

    def filter(self, scan):
        self.__compute_clusters(scan)

        # без ограничения на размер кластера – просто убираем шум
        if self.min_cluster_size is None:
            return [point for point in scan if getattr(point, "labels", -1) != -1]

        # считаем размер каждого кластера (кроме шума)
        cluster_counts = {}
        for point in scan:
            lbl = getattr(point, "labels", -1)
            if lbl == -1:
                continue
            cluster_counts[lbl] = cluster_counts.get(lbl, 0) + 1

        # кластеры, которые проходят порог
        large_clusters = {
            lbl for lbl, cnt in cluster_counts.items()
            if cnt >= self.min_cluster_size
        }

        # собираем точки только из достаточно больших кластеров
        point_lst = [
            point for point in scan
            if getattr(point, "labels", -1) in large_clusters
        ]

        return point_lst

if __name__ == "__main__":
    from app.base.scan.Scan import Scan
    from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL

    scan = Scan("l1")
    # scan.import_points_from_file(file_path=r"../../../../src/L1.las")
    scan.import_points_from_file(file_path=r"../../../../src/L1_raw.txt")
    # print(scan)

    # scan.plot(plotter=ScanPlotterWithLabelsMPL)

    for eps, min_samples, min_cluster_size in ((0.03, 100, 5000),
                                               # (0.02, 50, 1000),
                                               # (0.01, 25, 500),
                                               (0.007, 15, 250),
                                               (0.005, 10, 500),
                                               ):
        scan.filter_scan(filter_cls=ScanFilterByDBSCAN, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size)
        print(scan)
        scan.plot(plotter=ScanPlotterWithLabelsMPL)

