from abc import ABC, abstractmethod


class ScanPlotterABC(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self, scan):
        pass
