from CONFIG import DEFAULT_POINTS_COLOR
from app.base.Point import Point


class ScanPoint(Point):

    def __init__(self, x, y, z, color=DEFAULT_POINTS_COLOR, id_=None):
        super().__init__(x, y, z)
        self.color = color
        self.id_ = id_

    def __str__(self):
        if self.id_ is None:
            return f"ScanPoint (x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, color={self.color})"
        return f"ScanPoint (id={self.id_}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, color={self.color})"

    def __repr__(self):
        if self.id_ is None:
            return f"SP=[xyz=({self.x:.3f}, {self.y:.3f}, {self.z:.3f})])"
        return f"SP=[id={self.id_}, xyz=({self.x:.3f}, {self.y:.3f}, {self.z:.3f})])"
