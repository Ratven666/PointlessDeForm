import numpy as np
from sklearn.cluster import KMeans


class ScanNormalsDirectionClassifier:

    def __init__(self, scan):
        self.scan = scan

    def __scan_normals_to_numpy(self):
        # предполагается, что у каждого ScanPoint есть атрибут normals: np.ndarray shape (3,)
        normals = []
        for p in self.scan:
            n = getattr(p, "normals", None)
            if n is None:
                raise AttributeError("У точки нет normals, сначала вызови scan.compute_normals()")
            normals.append(n)
        return np.array(normals, dtype=float)

    @staticmethod
    def _unify_normals_hemisphere(normals):
        n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        ref = n.mean(axis=0)
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = ref / ref_norm
        dots = n @ ref
        mask = dots < 0
        n[mask] = -n[mask]
        return n

    def classify_normals(self, n_classes=3, unify_hemisphere=True):
        normals = self.__scan_normals_to_numpy()
        if unify_hemisphere:
            normals = self._unify_normals_hemisphere(normals)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
        labels = kmeans.fit_predict(normals)

        for p, lbl in zip(self.scan, labels):
            setattr(p, "labels", int(lbl))

        return labels, kmeans.cluster_centers_

if __name__ == "__main__":
    from app.base.scan.Scan import Scan
    from app.base.scan.plotters.ScanPlotterWithLabelsMPL import ScanPlotterWithLabelsMPL

    scan = Scan("l1")

    scan.import_points_from_file(file_path=r"../../../../src/L1.las")
    scan.compute_normals(k=8)
    s_dbscan_c = ScanNormalsDirectionClassifier(scan)
    s_dbscan_c.classify_normals(n_classes=3, unify_hemisphere=True)

    # for point in scan:
    #     print(point, point.labels)

    scan.plot(plotter=ScanPlotterWithLabelsMPL)