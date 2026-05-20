from app.base.Point import Point


class NamedPoint(Point):

    def __init__(self, name, x, y, z=0):
        super().__init__(x, y, z)
        self.name = name

    def __str__(self):
        return f"{self.__class__.__name__} (name={self.name}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    def __repr__(self):
        return f"({self. name} {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
