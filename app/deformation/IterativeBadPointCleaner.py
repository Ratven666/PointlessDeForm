from dataclasses import dataclass
from typing import List, Set

import pandas as pd

from app.cross_points.CrossPointSegmentSet import CrossPointSegmentSet

from app.deformation.RejectedLengthPointAnalyzer import RejectedLengthPointAnalyzer
from app.deformation.SegmentLengthDeformationAnalyzer import SegmentLengthDeformationAnalyzer


@dataclass
class CleaningIterationResult:
    iteration: int

    n_points_epoch1_before: int
    n_points_epoch2_before: int

    excluded_points_this_iter: list
    total_excluded_points: list

    n_points_epoch1_after: int
    n_points_epoch2_after: int

    n_segments_epoch1: int
    n_segments_epoch2: int

    n_deformation_results: int
    n_suspects: int
    n_highly_suspects: int

    stop_reason: str

    def as_dict(self):
        return {
            "iteration": self.iteration,
            "n_points_epoch1_before": self.n_points_epoch1_before,
            "n_points_epoch2_before": self.n_points_epoch2_before,
            "excluded_points_this_iter": ", ".join(self.excluded_points_this_iter),
            "total_excluded_points": ", ".join(self.total_excluded_points),
            "n_points_epoch1_after": self.n_points_epoch1_after,
            "n_points_epoch2_after": self.n_points_epoch2_after,
            "n_segments_epoch1": self.n_segments_epoch1,
            "n_segments_epoch2": self.n_segments_epoch2,
            "n_deformation_results": self.n_deformation_results,
            "n_suspects": self.n_suspects,
            "n_highly_suspects": self.n_highly_suspects,
            "stop_reason": self.stop_reason,
        }


@dataclass
class IterativeCleaningResult:
    cleaned_points_epoch1: list
    cleaned_points_epoch2: list

    excluded_points: list
    iterations: list

    last_deformation_results: list
    last_point_stats: list

    def iterations_to_dataframe(self):
        return pd.DataFrame([it.as_dict() for it in self.iterations])

    def point_stats_to_dataframe(self):
        return pd.DataFrame([s.as_dict() for s in self.last_point_stats])

    def deformation_results_to_dataframe(self):
        return pd.DataFrame([r.as_dict() for r in self.last_deformation_results])


class IterativeBadPointCleaner:
    """
    Итерационное исключение "плохих" точек по результатам анализа исключённых длин.

    Алгоритм:
    1. Строим сегменты по текущим наборам точек.
    2. Считаем деформационный анализ SegmentLengthDeformationAnalyzer.
    3. Находим подозрительные точки через RejectedLengthPointAnalyzer.
    4. Исключаем точки.
    5. Повторяем, пока не перестанут находиться новые плохие точки
       или пока не будет достигнут лимит итераций.

    Parameters
    ----------
    deformation_analyzer : SegmentLengthDeformationAnalyzer
        Анализатор изменений длин сегментов.
    rejected_point_analyzer : RejectedLengthPointAnalyzer
        Анализатор подозрительных точек по исключённым сегментам.
    remove_mode : str
        "highly_suspect" или "suspect"
    max_iterations : int
        Максимальное число итераций.
    min_points_left : int
        Минимально допустимое число точек, чтобы не вычистить всё.
    """

    def __init__(
        self,
        deformation_analyzer: SegmentLengthDeformationAnalyzer,
        rejected_point_analyzer: RejectedLengthPointAnalyzer,
        remove_mode="highly_suspect",
        max_iterations=10,
        min_points_left=5,
    ):
        if remove_mode not in {"highly_suspect", "suspect"}:
            raise ValueError("remove_mode must be 'highly_suspect' or 'suspect'")

        self.deformation_analyzer = deformation_analyzer
        self.rejected_point_analyzer = rejected_point_analyzer
        self.remove_mode = remove_mode
        self.max_iterations = max_iterations
        self.min_points_left = min_points_left

    def clean(self, points_epoch1, points_epoch2):
        current_points_epoch1 = list(points_epoch1)
        current_points_epoch2 = list(points_epoch2)

        excluded_points: Set[str] = set()
        iterations: List[CleaningIterationResult] = []

        last_deformation_results = []
        last_point_stats = []

        for iteration in range(1, self.max_iterations + 1):
            n1_before = len(current_points_epoch1)
            n2_before = len(current_points_epoch2)

            common_names = self._get_common_point_names(current_points_epoch1, current_points_epoch2)
            current_points_epoch1 = [p for p in current_points_epoch1 if p.name in common_names]
            current_points_epoch2 = [p for p in current_points_epoch2 if p.name in common_names]

            if len(current_points_epoch1) < self.min_points_left or len(current_points_epoch2) < self.min_points_left:
                iterations.append(CleaningIterationResult(
                    iteration=iteration,
                    n_points_epoch1_before=n1_before,
                    n_points_epoch2_before=n2_before,
                    excluded_points_this_iter=[],
                    total_excluded_points=sorted(excluded_points),
                    n_points_epoch1_after=len(current_points_epoch1),
                    n_points_epoch2_after=len(current_points_epoch2),
                    n_segments_epoch1=0,
                    n_segments_epoch2=0,
                    n_deformation_results=0,
                    n_suspects=0,
                    n_highly_suspects=0,
                    stop_reason=f"Too few points left (< {self.min_points_left})",
                ))
                break

            seg_set_1 = CrossPointSegmentSet.from_all_pairs(current_points_epoch1)
            seg_set_2 = CrossPointSegmentSet.from_all_pairs(current_points_epoch2)

            deformation_results = self.deformation_analyzer.analyze_for_all_points(seg_set_1, seg_set_2)
            point_stats = self.rejected_point_analyzer.analyze(deformation_results)

            last_deformation_results = deformation_results
            last_point_stats = point_stats

            suspects = [s for s in point_stats if s.is_suspect]
            highly_suspects = [s for s in point_stats if s.is_highly_suspect]

            if self.remove_mode == "highly_suspect":
                to_remove = sorted([s.point_name for s in highly_suspects if s.point_name not in excluded_points])
            else:
                to_remove = sorted([s.point_name for s in suspects if s.point_name not in excluded_points])

            if len(to_remove) == 0:
                iterations.append(CleaningIterationResult(
                    iteration=iteration,
                    n_points_epoch1_before=n1_before,
                    n_points_epoch2_before=n2_before,
                    excluded_points_this_iter=[],
                    total_excluded_points=sorted(excluded_points),
                    n_points_epoch1_after=len(current_points_epoch1),
                    n_points_epoch2_after=len(current_points_epoch2),
                    n_segments_epoch1=len(seg_set_1),
                    n_segments_epoch2=len(seg_set_2),
                    n_deformation_results=len(deformation_results),
                    n_suspects=len(suspects),
                    n_highly_suspects=len(highly_suspects),
                    stop_reason="No new suspect points found",
                ))
                break

            future_n1 = len([p for p in current_points_epoch1 if p.name not in to_remove])
            future_n2 = len([p for p in current_points_epoch2 if p.name not in to_remove])

            if future_n1 < self.min_points_left or future_n2 < self.min_points_left:
                iterations.append(CleaningIterationResult(
                    iteration=iteration,
                    n_points_epoch1_before=n1_before,
                    n_points_epoch2_before=n2_before,
                    excluded_points_this_iter=to_remove,
                    total_excluded_points=sorted(excluded_points | set(to_remove)),
                    n_points_epoch1_after=len(current_points_epoch1),
                    n_points_epoch2_after=len(current_points_epoch2),
                    n_segments_epoch1=len(seg_set_1),
                    n_segments_epoch2=len(seg_set_2),
                    n_deformation_results=len(deformation_results),
                    n_suspects=len(suspects),
                    n_highly_suspects=len(highly_suspects),
                    stop_reason=f"Stopping to avoid dropping below min_points_left={self.min_points_left}",
                ))
                break

            excluded_points.update(to_remove)

            current_points_epoch1 = [p for p in current_points_epoch1 if p.name not in excluded_points]
            current_points_epoch2 = [p for p in current_points_epoch2 if p.name not in excluded_points]

            iterations.append(CleaningIterationResult(
                iteration=iteration,
                n_points_epoch1_before=n1_before,
                n_points_epoch2_before=n2_before,
                excluded_points_this_iter=to_remove,
                total_excluded_points=sorted(excluded_points),
                n_points_epoch1_after=len(current_points_epoch1),
                n_points_epoch2_after=len(current_points_epoch2),
                n_segments_epoch1=len(seg_set_1),
                n_segments_epoch2=len(seg_set_2),
                n_deformation_results=len(deformation_results),
                n_suspects=len(suspects),
                n_highly_suspects=len(highly_suspects),
                stop_reason="Continue",
            ))

        return IterativeCleaningResult(
            cleaned_points_epoch1=current_points_epoch1,
            cleaned_points_epoch2=current_points_epoch2,
            excluded_points=sorted(excluded_points),
            iterations=iterations,
            last_deformation_results=last_deformation_results,
            last_point_stats=last_point_stats,
        )

    @staticmethod
    def _get_common_point_names(points_epoch1, points_epoch2):
        names1 = {p.name for p in points_epoch1}
        names2 = {p.name for p in points_epoch2}
        return names1 & names2
