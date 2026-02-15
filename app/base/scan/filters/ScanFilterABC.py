from abc import ABC, abstractmethod


class ScanFilterABC(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def filter(self, scan):
        pass
