from matplotlib import pyplot as plt

from app.base.scan.plotters.ScanPlotterABC import ScanPlotterABC


class ScanPlotterMPL(ScanPlotterABC):

    def __init__(self, fig_ax=None, is_show=True):
        self.fig, self.ax = fig_ax if fig_ax is not None else (None, None)
        self.is_show = is_show

    def plot(self, scan):
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
        x, y, z, c = [], [], [], []
        for point in scan:
            x.append(point.x)
            y.append(point.y)
            z.append(point.z)
            rgb = [rgb / 255 for rgb in point.color]
            c.append(rgb)
        self.ax.scatter(x, y, z, c=c)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.axis('equal')
        if self.is_show:
            plt.show()
        return self.fig, self.ax
