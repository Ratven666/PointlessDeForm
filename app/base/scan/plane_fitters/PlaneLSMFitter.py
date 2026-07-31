import numpy as np

from app.base.scan.plane_fitters.PlaneFitterABC import PlaneFitterABC


class PlaneLSMFitter(PlaneFitterABC):
    """
    МНК-подгонка плоскости через PCA (SVD центрированных точек).

    Модель:  n^T * x + d = 0,  где ||n|| = 1.

    После подгонки сохраняет ковариационную матрицу параметров (A,B,C,D) 4×4,
    доступную как атрибут `cov_params`, и оценку единичного веса `sigma0`.

    Вывод ковариации:
    -----------------
    Подгонка эквивалентна МНК в форме:
        n^T * pts[i] + d = v[i],  E[v[i]] = 0,  D[v[i]] = sigma^2
    Матрица дизайна  X  (N×4): каждая строка — [xi, yi, zi, 1].
    Нормаль — нулевой (минимальный) сингулярный вектор X (с нулевым средним по d),
    то есть минимальный собственный вектор X^T X.

    Ковариация вектора параметров p = (A,B,C,D):
        Sigma_p = sigma0^2 * (X^T X)^{-1}
    где  sigma0^2 = sum(v_i^2) / (N - 4)  — несмещённая оценка дисперсии остатков.

    Примечание о нормировке нормали:
        Поскольку нормаль единичная, (A,B,C,D) — это не четыре независимых параметра,
        а параметр D является зависимым. Данная ковариация является приближённой (первый
        порядок) и корректна для оценки ошибок при пропагации через линейную систему.
    """

    def fit_plane(self, *args, **kwargs):
        pts = self._scan_to_numpy()
        n_pts = pts.shape[0]
        if n_pts < 4:
            raise ValueError("Для оценки ковариации нужно минимум 4 точки")

        centroid = pts.mean(axis=0)
        centered = pts - centroid

        _, sv, vh = np.linalg.svd(centered, full_matrices=False)

        # нормаль — последний строчный вектор VH (минимальное сингулярное значение)
        normal = vh[-1, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)

        point_on_plane = centroid

        # --- ковариационная матрица параметров (A,B,C,D) ---
        # Матрица дизайна: [x, y, z, 1] для каждой точки
        ones = np.ones((n_pts, 1), dtype=float)
        X = np.hstack([pts, ones])          # (N, 4)
        XtX = X.T @ X                       # (4, 4)

        # невязки: расстояния со знаком (нормаль единичная)
        residuals = pts @ normal + d        # (N,)

        # несмещённая оценка дисперсии единицы веса
        dof = n_pts - 4
        if dof <= 0:
            self.sigma0 = 0.0
            self.cov_params = np.zeros((4, 4), dtype=float)
        else:
            sigma0_sq = float(np.sum(residuals ** 2) / dof)
            self.sigma0 = float(np.sqrt(sigma0_sq))

            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)

            self.cov_params = sigma0_sq * XtX_inv   # (4, 4)

        scan = self.scan
        return scan, normal, point_on_plane, d
