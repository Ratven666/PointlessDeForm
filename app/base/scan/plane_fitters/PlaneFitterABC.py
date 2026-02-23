from abc import ABC, abstractmethod

import numpy as np

from app.base.scan.Scan import Scan


class PlaneFitterABC(ABC):

    def __init__(self, scan: Scan):
        self.scan = scan


    def _scan_to_numpy(self):
        """Все точки скана в np.ndarray (N,3)."""
        pts = np.array([[p.x, p.y, p.z] for p in self.scan], dtype=float)
        return pts

    @abstractmethod
    def fit_plane(self, *args, **kwargs):
        pass