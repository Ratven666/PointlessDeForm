from itertools import combinations

import numpy as np
import pandas as pd

from app.util.CrossPointSegment import CrossPointSegment


class CrossPointSegmentSet:
    """
    Набор отрезков CrossPointSegment, построенных по списку точек CrossPoint.

    Возможности:
    - построение всех попарных отрезков;
    - построение отрезков по явному списку пар имён;
    - получение отрезка по именам точек;
    - экспорт в DataFrame;
    - базовая статистика по длинам и погрешностям.
    """

    def __init__(self, segments=None, name="CrossPointSegmentSet"):
        self.name = name
        self._segments = list(segments) if segments is not None else []

    # ------------------------------------------------------------------
    # Магические методы контейнера
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._segments)

    def __iter__(self):
        return iter(self._segments)

    def __getitem__(self, item):
        return self._segments[item]

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, n_segments={len(self)})"

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    # Добавление
    # ------------------------------------------------------------------
    def add_segment(self, segment: CrossPointSegment):
        if not isinstance(segment, CrossPointSegment):
            raise TypeError(
                f"segment должен быть CrossPointSegment, получено: {type(segment).__name__}"
            )
        self._segments.append(segment)
        return self

    # ------------------------------------------------------------------
    # Фабрики
    # ------------------------------------------------------------------
    @classmethod
    def from_all_pairs(cls, points, cross_cov_map=None, name="AllPairsSegmentSet"):
        """
        Строит все возможные попарные отрезки из списка точек.

        Parameters
        ----------
        points : list[CrossPoint]
        cross_cov_map : dict | None
            Словарь взаимных ковариаций:
            {
                ("P1", "P2"): cov12,
                ("P2", "P1"): cov21,   # можно не дублировать
            }
        """
        seg_set = cls(name=name)

        for p1, p2 in combinations(points, 2):
            cross_cov = cls._get_cross_cov(cross_cov_map, p1.name, p2.name)
            seg = CrossPointSegment(p1, p2, cross_cov_12=cross_cov)
            seg_set.add_segment(seg)

        return seg_set

    @classmethod
    def from_named_pairs(cls, points, pairs, cross_cov_map=None, name="NamedPairsSegmentSet"):
        """
        Строит отрезки по заданному списку пар имён.

        Parameters
        ----------
        points : list[CrossPoint]
        pairs : iterable[tuple[str, str]]
            Например: [("P1", "P2"), ("P2", "P5")]
        cross_cov_map : dict | None
            Словарь взаимных ковариаций между точками.
        """
        point_dict = {p.name: p for p in points}
        seg_set = cls(name=name)

        for name1, name2 in pairs:
            if name1 not in point_dict:
                raise KeyError(f"Точка '{name1}' не найдена")
            if name2 not in point_dict:
                raise KeyError(f"Точка '{name2}' не найдена")
            if name1 == name2:
                raise ValueError(f"Нельзя построить отрезок между одинаковыми точками: {name1}")

            p1 = point_dict[name1]
            p2 = point_dict[name2]
            cross_cov = cls._get_cross_cov(cross_cov_map, name1, name2)

            seg = CrossPointSegment(p1, p2, cross_cov_12=cross_cov)
            seg_set.add_segment(seg)

        return seg_set

    # ------------------------------------------------------------------
    # Поиск и фильтрация
    # ------------------------------------------------------------------
    def get_by_names(self, name1, name2):
        """
        Возвращает отрезок между двумя точками независимо от порядка имён.
        """
        target = {name1, name2}
        for seg in self._segments:
            if {seg.p1.name, seg.p2.name} == target:
                return seg
        raise KeyError(f"Отрезок между '{name1}' и '{name2}' не найден")

    def filter_by_names(self, point_names):
        """
        Оставляет только те отрезки, у которых обе точки принадлежат point_names.
        """
        point_names = set(point_names)
        segments = [
            seg for seg in self._segments
            if seg.p1.name in point_names and seg.p2.name in point_names
        ]
        return CrossPointSegmentSet(segments=segments, name=f"{self.name}_filtered")

    def filter_reliable(self):
        """
        Оставляет только отрезки с reliable_accuracy=True.
        """
        segments = [seg for seg in self._segments if seg.reliable_accuracy]
        return CrossPointSegmentSet(segments=segments, name=f"{self.name}_reliable")

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------
    def as_list_of_dicts(self):
        return [seg.as_dict() for seg in self._segments]

    def to_dataframe(self):
        """
        Экспорт набора отрезков в pandas.DataFrame.
        """
        return pd.DataFrame(self.as_list_of_dicts())

    def to_csv(self, file_path, index=False):
        df = self.to_dataframe()
        df.to_csv(file_path, index=index)

    def to_excel(self, file_path, index=False):
        df = self.to_dataframe()
        df.to_excel(file_path, index=index)

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------
    def get_length_array(self):
        return np.array([seg.length for seg in self._segments], dtype=float)

    def get_sigma_length_array(self):
        vals = [seg.sigma_length for seg in self._segments if seg.sigma_length is not None]
        if not vals:
            return np.array([], dtype=float)
        return np.array(vals, dtype=float)

    def summary(self):
        """
        Возвращает краткую статистику по набору отрезков.
        """
        if len(self) == 0:
            return {
                "name": self.name,
                "n_segments": 0,
                "n_reliable": 0,
                "length_min": None,
                "length_max": None,
                "length_mean": None,
                "sigma_length_mean": None,
                "sigma_length_max": None,
            }

        lengths = self.get_length_array()
        sigma_lengths = self.get_sigma_length_array()
        n_reliable = sum(seg.reliable_accuracy for seg in self._segments)

        return {
            "name": self.name,
            "n_segments": len(self),
            "n_reliable": n_reliable,
            "length_min": float(np.min(lengths)),
            "length_max": float(np.max(lengths)),
            "length_mean": float(np.mean(lengths)),
            "sigma_length_mean": None if len(sigma_lengths) == 0 else float(np.mean(sigma_lengths)),
            "sigma_length_max": None if len(sigma_lengths) == 0 else float(np.max(sigma_lengths)),
        }

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    @staticmethod
    def _get_cross_cov(cross_cov_map, name1, name2):
        """
        Возвращает взаимную ковариацию Cov(X1, X2), если она есть.
        Допускаются ключи как (name1, name2), так и (name2, name1).
        """
        if cross_cov_map is None:
            return None

        if (name1, name2) in cross_cov_map:
            return cross_cov_map[(name1, name2)]

        if (name2, name1) in cross_cov_map:
            cov = np.asarray(cross_cov_map[(name2, name1)], dtype=float)
            return cov.T

        return None


if __name__ == "__main__":
    from app.util.CrossPointListRestorer import CrossPointListRestorer

    points_path = "/Users/mikhail_vystrchil/Documents/MY_PROGRAMMS/PointlessDeForm/data/8_floors_wall/output/scan_2334_filt/cross_points_good_filtered_by_ellipsoid.csv"

    restorer = CrossPointListRestorer(points_path)
    points = restorer.restore_all()

    seg_set_all = CrossPointSegmentSet.from_all_pairs(points)
    print(seg_set_all)
    print(seg_set_all.summary())

    df = seg_set_all.to_dataframe()
    print(df.head())

    seg = seg_set_all.get_by_names(points[0].name, points[1].name)
    print(seg)

    pairs = [
        (points[0].name, points[1].name),
        (points[1].name, points[2].name),
    ]
    seg_set_named = CrossPointSegmentSet.from_named_pairs(points, pairs)
    print(seg_set_named.summary())
