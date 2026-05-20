from app.base.NamedPoint import NamedPoint


class CrossPoint(NamedPoint):

    def __init__(self, name, x, y, z=0):
        super().__init__(name, x, y, z)
        self.status: str|None = None
        self.mse: float|None = None
        self.planes_mse: list[float]|None = None

    def load_mses(self, plane_mses:list[float]):
        self.planes_mse = plane_mses
        self.mse = sum([mse**2 for mse in plane_mses]) ** 0.5

    def __str__(self):
        return (f"{self.__class__.__name__} (name={self.name}, status={self.status}, "
                f"x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
                f"plane_mses={str([round(mse, 5) for mse in self.planes_mse])}, mse={self.mse:.5f})")

    def __repr__(self):
        return (f"({self. name}, status={self.status}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f}, "
                f"mse={self.mse:.5f})")
