from dataclasses import dataclass
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd




@dataclass
class RejectedPointStats:
    """
    Статистика подозрительности точки по исключённым длинам.
    """

    point_name: str

    n_total_links: int
    n_rejected_links: int
    rejection_ratio: float

    rejected_with_points: list
    source_center_points: list

    suspicion_score: float
    is_suspect: bool
    is_highly_suspect: bool

    def as_dict(self):
        return {
            "point_name": self.point_name,
            "n_total_links": self.n_total_links,
            "n_rejected_links": self.n_rejected_links,
            "rejection_ratio": self.rejection_ratio,
            "rejected_with_points": ", ".join(sorted(self.rejected_with_points)),
            "source_center_points": ", ".join(sorted(self.source_center_points)),
            "suspicion_score": self.suspicion_score,
            "is_suspect": self.is_suspect,
            "is_highly_suspect": self.is_highly_suspect,
        }

    def __str__(self):
        level = "HIGHLY_SUSPECT" if self.is_highly_suspect else ("SUSPECT" if self.is_suspect else "ok")
        return (
            f"RejectedPointStats(point={self.point_name}, "
            f"rejected={self.n_rejected_links}/{self.n_total_links}, "
            f"ratio={self.rejection_ratio:.3f}, "
            f"score={self.suspicion_score:.3f}, "
            f"status={level})"
        )


class RejectedLengthPointAnalyzer:
    """
    Анализирует результаты SegmentLengthDeformationAnalyzer и выделяет
    подозрительные точки по исключённым сегментам.

    Логика:
    - каждый исключённый сегмент "central_point -- neighbor_point"
      добавляет по одному инциденту подозрительности обеим точкам;
    - затем по каждой точке считаются:
        * число всех связей,
        * число исключённых связей,
        * доля исключений,
        * интегральный score.

    Parameters
    ----------
    min_rejected_links_for_suspect : int
        Минимальное число исключённых связей, чтобы точка стала suspect.
    rejection_ratio_for_suspect : float
        Минимальная доля исключённых связей, чтобы точка стала suspect.
    min_rejected_links_for_highly_suspect : int
        Минимальное число исключённых связей для highly_suspect.
    rejection_ratio_for_highly_suspect : float
        Минимальная доля исключённых связей для highly_suspect.
    """

    def __init__(
        self,
        min_rejected_links_for_suspect=3,
        rejection_ratio_for_suspect=0.25,
        min_rejected_links_for_highly_suspect=5,
        rejection_ratio_for_highly_suspect=0.40,
    ):
        self.min_rejected_links_for_suspect = min_rejected_links_for_suspect
        self.rejection_ratio_for_suspect = rejection_ratio_for_suspect
        self.min_rejected_links_for_highly_suspect = min_rejected_links_for_highly_suspect
        self.rejection_ratio_for_highly_suspect = rejection_ratio_for_highly_suspect

    def analyze(self, deformation_results) -> List[RejectedPointStats]:
        """
        Parameters
        ----------
        deformation_results : list[LengthDeformationResult]
            Результаты из SegmentLengthDeformationAnalyzer.analyze_for_all_points(...)
        """
        total_links = defaultdict(set)
        rejected_links = defaultdict(set)
        source_centers = defaultdict(set)

        for result in deformation_results:
            center = result.point_name

            # Все использованные связи центральной точки
            for neighbor in getattr(result, "used_neighbors", []):
                total_links[center].add(neighbor)
                total_links[neighbor].add(center)

            # Исключённые связи
            for neighbor in getattr(result, "rejected_neighbors", []):
                rejected_links[center].add(neighbor)
                rejected_links[neighbor].add(center)

                source_centers[center].add(center)
                source_centers[neighbor].add(center)

                # Даже если связь была исключена, она всё равно является связью точки
                total_links[center].add(neighbor)
                total_links[neighbor].add(center)

        all_points = sorted(set(total_links.keys()) | set(rejected_links.keys()) | set(source_centers.keys()))

        stats_list = []
        for point_name in all_points:
            n_total = len(total_links[point_name])
            n_rejected = len(rejected_links[point_name])

            if n_total > 0:
                ratio = n_rejected / n_total
            else:
                ratio = 0.0

            # Простой интегральный рейтинг:
            # половина веса — абсолютное число,
            # половина — относительная доля.
            score = 0.5 * n_rejected + 0.5 * ratio

            is_suspect = (
                n_rejected >= self.min_rejected_links_for_suspect
                or ratio >= self.rejection_ratio_for_suspect
            )

            is_highly_suspect = (
                n_rejected >= self.min_rejected_links_for_highly_suspect
                or ratio >= self.rejection_ratio_for_highly_suspect
            )

            stats = RejectedPointStats(
                point_name=point_name,
                n_total_links=n_total,
                n_rejected_links=n_rejected,
                rejection_ratio=ratio,
                rejected_with_points=sorted(rejected_links[point_name]),
                source_center_points=sorted(source_centers[point_name]),
                suspicion_score=score,
                is_suspect=is_suspect,
                is_highly_suspect=is_highly_suspect,
            )
            stats_list.append(stats)

        stats_list.sort(
            key=lambda x: (x.is_highly_suspect, x.is_suspect, x.suspicion_score, x.n_rejected_links),
            reverse=True
        )

        return stats_list

    @staticmethod
    def to_dataframe(stats_list):
        return pd.DataFrame([s.as_dict() for s in stats_list])

    @staticmethod
    def filter_only_suspects(stats_list):
        return [s for s in stats_list if s.is_suspect]

    @staticmethod
    def filter_only_highly_suspects(stats_list):
        return [s for s in stats_list if s.is_highly_suspect]


if __name__ == "__main__":
    from app.util.CrossPointListRestorer import CrossPointListRestorer
    from app.util.CrossPointSegmentSet import CrossPointSegmentSet
    from app.util.LengthDeformationResult import SegmentLengthDeformationAnalyzer

    points_path_epoch1 = "/Users/mikhail_vystrchil/Documents/MY_PROGRAMMS/PointlessDeForm/data/8_floors_wall/output/total_scan/cross_points_good_filtered_by_ellipsoid.csv"
    points_path_epoch2 = "/Users/mikhail_vystrchil/Documents/MY_PROGRAMMS/PointlessDeForm/data/8_floors_wall/output/scan_2335_filt/cross_points_good_filtered_by_ellipsoid.csv"

    points1 = CrossPointListRestorer(points_path_epoch1).restore_all()
    points2 = CrossPointListRestorer(points_path_epoch2).restore_all()

    seg_set_1 = CrossPointSegmentSet.from_all_pairs(points1)
    seg_set_2 = CrossPointSegmentSet.from_all_pairs(points2)

    deformation_analyzer = SegmentLengthDeformationAnalyzer(
        alpha=0.05,
        use_weights=True,
        enable_rejection=True,
        rejection_threshold=3.0,
        min_obs=3,
        max_rejections=10,
    )

    deformation_results = deformation_analyzer.analyze_for_all_points(seg_set_1, seg_set_2)

    rejected_point_analyzer = RejectedLengthPointAnalyzer(
        min_rejected_links_for_suspect=3,
        rejection_ratio_for_suspect=0.25,
        min_rejected_links_for_highly_suspect=5,
        rejection_ratio_for_highly_suspect=0.40,
    )

    point_stats = rejected_point_analyzer.analyze(deformation_results)

    for s in point_stats:
        print(s)

    df = rejected_point_analyzer.to_dataframe(point_stats)
    print(df.head(20))

    suspects = rejected_point_analyzer.filter_only_suspects(point_stats)
    print("\nSUSPECT POINTS:")
    for s in suspects:
        print(s)