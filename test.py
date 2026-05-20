import numpy as np

def has_parallel_subset(planes, angle_tol_rad=np.deg2rad(1.0), min_group_size=2):
    """
    planes          – iterable[Plane]
    angle_tol_rad   – допуск по углу между нормалями (радианы)
    min_group_size  – минимальный размер подмножества параллельных плоскостей
    Возвращает True, если существует подмножество из min_group_size
    (или больше) плоскостей, которые попарно параллельны в пределах допуска.
    """
    planes = list(planes)
    n = len(planes)
    if n < min_group_size:
        return False
    normals = np.array([p.normal for p in planes], dtype=float)
    # нормируем нормали на всякий случай
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    # косинус допуска
    cos_tol = np.cos(angle_tol_rad)
    # пытаемся сгруппировать нормали по направлению (с учётом ±)
    groups = []
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        # создаём новую группу с базовой нормалью
        base = normals[i]
        group = [i]
        used[i] = True
        # ищем все нормали, параллельные base в пределах допуска
        dots = normals @ base  # скалярные произведения
        # параллельность с учётом направления: |cos(theta)| ≈ 1
        mask = np.abs(dots) >= cos_tol
        idxs = np.where(mask & (~used))[0]
        for j in idxs:
            group.append(j)
            used[j] = True
        if len(group) >= min_group_size:
            return True
    return False

def parallel_groups(planes, angle_tol_rad=np.deg2rad(1.0), min_group_size=2, return_indices=False):
    """
    Возвращает список групп параллельных плоскостей.
    Каждая группа – список плоскостей (или индексов, если return_indices=True).
    """
    planes = list(planes)
    n = len(planes)
    if n < min_group_size:
        return []
    normals = np.array([p.normal for p in planes], dtype=float)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    cos_tol = np.cos(angle_tol_rad)
    groups = []
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        base = normals[i]
        group = [i]
        dots = normals @ base
        mask = np.abs(dots) >= cos_tol
        idxs = np.where(mask & (~used))[0]
        for j in idxs:
            group.append(j)
        if len(group) >= min_group_size:
            for j in group:
                used[j] = True
            if return_indices:
                groups.append(group)
            else:
                groups.append([planes[j] for j in group])
    return groups


planes = [Plane(...), Plane(...), ...]
if has_parallel_subset(planes, angle_tol_rad=np.deg2rad(2), min_group_size=3):
    print("Есть хотя бы три параллельные плоскости в пределах 2°.")
else:
    print("Нет такого подмножества.")