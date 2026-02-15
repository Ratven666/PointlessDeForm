import numpy as np
from matplotlib import pyplot as plt

from app.base.scan.plotters.ScanPlotterMPL import ScanPlotterMPL


class ScanPlotterWithLabelsMPL(ScanPlotterMPL):

    def plot(self, scan):
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
        try:
            X = np.array([[p.x, p.y, p.z, p.labels] for p in scan])
        except AttributeError:
            print("Скан не классифицирован!")
            super().plot(scan)
            return

        x, y, z, labels = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        unique_labels = np.unique(labels)
        # colors = plt.cm.get_cmap("tab20", len(unique_labels))

        # шум отдельно, остальные кластеры нумеруем с 0
        cluster_labels = [lbl for lbl in unique_labels if lbl != -1]
        n_clusters = len(cluster_labels)

        cmap = plt.cm.get_cmap("tab20", n_clusters if n_clusters > 0 else 1)
        label_to_color_idx = {lbl: i for i, lbl in enumerate(cluster_labels)}

        for k in unique_labels:
            mask = labels == k
            if k == -1:
                self.ax.scatter(x[mask], y[mask], z[mask],
                                s=1, c='k', alpha=0.1, label='noise')
            else:
                ci = label_to_color_idx[k]  # индекс 0..n_clusters-1
                self.ax.scatter(x[mask], y[mask], z[mask],
                                s=5, color=cmap(ci), label=f'cluster {k}')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend(loc='best', markerscale=4)
        plt.tight_layout()
        plt.axis('equal')
        if self.is_show:
            plt.show()
        return self.fig, self.ax


if __name__ == "__main__":
    from app.base.scan.Scan import Scan
    from app.base.scan.utils.ScanDBSCANClusterizator import ScanDBSCANClusterizator

    scan = Scan("l1")
    # print(scan)
    # scan.import_points_from_file(file_path=r"../../../../src/L1.las")
    scan.import_points_from_file(file_path=r"../../../../src/L1_raw.txt")
    # print(scan)

    # scan.plot(plotter=ScanPlotterWithLabelsMPL)

    s_dbscan_c = ScanDBSCANClusterizator(scan)
    s_dbscan_c.compute_clusters(eps=0.05, min_samples=100)

    scan.plot(plotter=ScanPlotterWithLabelsMPL)

