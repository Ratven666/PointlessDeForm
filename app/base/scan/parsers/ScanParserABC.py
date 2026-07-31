from abc import ABC, abstractmethod


class ScanParserABC(ABC):

    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def parse(self, scan):
        pass
