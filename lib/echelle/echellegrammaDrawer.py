from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import multiprocessing as mp
from  lib.echelle.dataClasses import (
    OrderEdges, Spectrometer
)
from lib.echelle.echelleMath import (
    lambda_range, diffraction_angle,
    find_order_edges,
    prism_incidence_angle_rad,
    grating_groove_tilt_rad
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class RawOrderEdges:
    """Сырые границы, возвращаемые воркером (без обрезки)."""
    k: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def _clip_to_detector(xs: np.ndarray, ys: np.ndarray, matrix_size: float, dx: float, dy: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Обрезать массивы координат по области детектора.
    xs, ys — массивы shape (N,2).
    Возвращает x_clipped, y_clipped (те же формы).
    Параметры обрезки взяты в соответствии с вашей прежней логикой:
      X ∈ [-matrix_size/2 + dx, +matrix_size/2 + dx]
      Y ∈ [0 + dy, matrix_size + dy]
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.ndim == 1:
        xs = xs.reshape(1, 2)
    if ys.ndim == 1:
        ys = ys.reshape(1, 2)

    half = float(matrix_size) / 2.0
    x_min_lim = -half + float(dx)
    x_max_lim = half - float(dx)
    y_min_lim = 0.0 + float(dy)
    y_max_lim = float(matrix_size) - float(dy)

    # Маска: True — если хотя бы одна координата попадает в допустимый диапазон
    valid_mask = np.logical_or(
        np.logical_and(ys[:, 0] >= y_min_lim, ys[:, 0] <= y_max_lim),
        np.logical_and(ys[:, 1] >= y_min_lim, ys[:, 1] <= y_max_lim)
    )

    # Применяем маску
    xs_filtered = xs[valid_mask]
    ys_filtered = ys[valid_mask]

    if ys_filtered[0][1] > y_max_lim:
        dx = xs_filtered[0][1] - xs_filtered[0][0]
        dy = ys_filtered[0][1] - ys_filtered[0][0]
        k = dy/dx

        dy_clipped = y_max_lim - ys_filtered[0][0]

        xs_filtered[0][1] = xs_filtered[0][0] + dy_clipped/k
        ys_filtered[0][1] = y_max_lim

    x_clipped = np.clip(xs_filtered, x_min_lim, x_max_lim)
    y_clipped = np.clip(ys_filtered, y_min_lim, y_max_lim)
    return x_clipped, y_clipped, valid_mask


def evaluate_k_worker(
    k: int,
    prism_wedge_angle_rad: float,
    prism_incidence_angle_rad: float,
    grating_groove_tilt_rad: float,
    focal: float,
    lines_in_mm: float,
    gamma_rad: float,
    grating_cross_tilt_rad: float,
    df_avg: float,
    df_prism_min: float,
    glass_type: str,
    dx: float,
    dy: float,
) -> Optional[RawOrderEdges]:
    """
    Воркер: вычисляет сырые границы одного порядка (без обрезки/выравнивания).
    Возвращает RawOrderEdges или None при ошибке.
    """
    try:
        x_min, y_min, x_max, y_max, _, _ = find_order_edges(
            k,
            prism_wedge_angle_rad,
            prism_incidence_angle_rad,
            grating_groove_tilt_rad,
            focal,
            lines_in_mm,
            gamma_rad,
            grating_cross_tilt_rad,
            df_avg,
            df_prism_min,
            glass_type,
        )

        return RawOrderEdges(k=int(k),
                             x_min=float(x_min + dx),
                             x_max=float(x_max + dx),
                             y_min=float(y_min + dy),
                             y_max=float(y_max + dy))
    except Exception as exc:
        logger.exception("Ошибка при вычислении краёв для k=%s: %s", k, exc)
        return None


class EchellegrammaDrawer:
    def __init__(self,
                 spectrometer: Spectrometer,
                 matrix_size: float, dx: float = None, dy: float = None):
        """

        :param lines_in_mm: количество штрихов на миллиметр дифракционной решетки
        :param gamma_deg: угол блеска дифракционной решетки в градусах
        :param gr_cross_tilt_deg: угол наклона решетки поперек штриха
        :param prism_tilt_deg: угол наклона призмы из Земакса (сохранять знак)
        :param prism_wedge_angle_deg: угол клина призмы
        :param grating_tilt_deg: угол наклона решетки из Земакса (сохранять знак)
        :param k_min: минимальный порядок дифракции
        :param k_max: максимальный порядок дифракции
        :param focal: фокусное расстояние
        :param matrix_size: размер детектора
        :param glass_type: материал призмы (CaF, BaF)
        :param df_avg: средняя дисперсия решетки для сдвига
        :param df_prism_min: минимальная дисперсия призмы для сдвига
        :param dx: сдвиг эшеллеграммы по X
        :param dy: сдвиг эшеллеграммы по Y
        """
        self.lines_in_mm = spectrometer.result.N
        self.gamma_rad = spectrometer.grating.gamma_rad
        self.grating_cross_tilt_rad = spectrometer.grating.gr_cross_tilt_rad
        self.k_min = spectrometer.result.kmin
        self.k_max = spectrometer.result.kmax
        self.focal = spectrometer.result.f_mm
        self.prism_wedge_angle_rad = math.radians(spectrometer.result.prism_deg)
        self.glass_type = spectrometer.prism.glass_type
        self.gap_mm = spectrometer.result.gap_mm
        self.df_avg = spectrometer.df_avg
        self.df_prism_min = spectrometer.df_prism_min
        self.matrix_size = matrix_size
        self.grating_tilt_deg = spectrometer.grating.grating_tilt_deg
        self.prism_tilt_deg = spectrometer.prism.tilt_deg
        # угол падения на призму
        self.prism_incidence_angle_rad = prism_incidence_angle_rad(
            prism_tilt_deg=self.prism_tilt_deg,
            prism_wedge_angle_rad=float(self.prism_wedge_angle_rad)
        )

        # угол наклона решетки вдоль штриха
        self.grating_groove_tilt_rad = grating_groove_tilt_rad(
            prism_wedge_angle_rad=float(self.prism_wedge_angle_rad),
            prism_tilt_deg=self.prism_tilt_deg,
            grating_tilt_deg=self.grating_tilt_deg
        )
        self.dx = 0 if dx is None else dx
        self.dy = 0 if dy is None else dy

    def draw_echelegramma(self, use_multiprocessing: bool = True) -> List[OrderEdges]:
        """
        Собирает все порядки, выравнивает эшеллеграмму по правилам (по X — среднее -> 0,
        по Y — y_min максимального порядка -> dy), делает обрезку и возвращает список OrderEdges.
        """
        if self.k_max < self.k_min:
            raise ValueError("k_max should be >= k_min")

        k_range = list(range(self.k_min, self.k_max + 1))

        # Подготовим аргументы для воркера (order-preserving)
        worker_args = [
            (
                int(k),
                float(self.prism_wedge_angle_rad),
                float(self.prism_incidence_angle_rad),
                float(self.grating_groove_tilt_rad),
                float(self.focal),
                float(self.lines_in_mm),
                float(self.gamma_rad),
                float(self.grating_cross_tilt_rad),
                float(self.df_avg),
                float(self.df_prism_min),
                str(self.glass_type),
                float(self.dx),
                float(self.dy),
            )
            for k in k_range
        ]

        # Запуск воркеров
        if use_multiprocessing and len(worker_args) > 1:
            cpu_count = max(1, mp.cpu_count())
            with mp.Pool(processes=cpu_count) as pool:
                results = pool.starmap(evaluate_k_worker, worker_args)
        else:
            results = [evaluate_k_worker(*args) for args in worker_args]

        # Фильтруем None и сортируем по порядку k
        raw_list: List[RawOrderEdges] = [r for r in results if r is not None]
        if not raw_list:
            logger.warning("Нет валидных порядков — возвращаю пустой список.")
            return []

        raw_list.sort(key=lambda r: r.k)

        # Формируем массивы shape (N, 2): [x_min, x_max], [y_min, y_max]
        xs = np.array([[r.x_min, r.x_max] for r in raw_list], dtype=float)
        ys = np.array([[r.y_min, r.y_max] for r in raw_list], dtype=float)

        # 1) Центруем по X: среднее значение по центрам порядков -> 0
        centers_x = (xs[:, 0] + xs[:, 1]) / 2.0
        x_center_mean = float(np.mean(centers_x)) # + self.dx
        xs_centered = xs #- x_center_mean  # broadcasting

        # 2) Находим "максимальный порядок" — тот, у которого ширина по X минимальна
        widths = np.abs(xs_centered[:, 1] - xs_centered[:, 0])
        idx_max_order = int(np.argmax(widths))

        # 3) Смещение по Y: y_min выбранного порядка должен стать self.dy
        y_shift = ys[idx_max_order, 1]
        ys_shifted = ys #+ y_shift

        # 4) Переворачиваем по Y
        ys_shifted = float(np.max(ys_shifted)) - ys_shifted + 0 / 2

        # 4) Обрезка по детектору (векторно)
        x_clipped, y_clipped, valid_mask = _clip_to_detector(xs_centered, ys_shifted,
                                                 matrix_size=self.matrix_size,
                                                 dx=0, dy=0)

        # Фильтруем индексы в raw_list
        filtered_raw_list = [raw for raw, valid in zip(raw_list, valid_mask) if valid]

        # Формируем итоговый список OrderEdges
        output: List[OrderEdges] = []
        for i, raw in enumerate(filtered_raw_list):
            oe = OrderEdges(
                k=raw.k,
                x_min=float(xs_centered[i, 0]),
                x_max=float(xs_centered[i, 1]),
                y_min=float(ys_shifted[i, 0]),
                y_max=float(ys_shifted[i, 1]),
                x_min_clipped=float(x_clipped[i, 0]),
                x_max_clipped=float(x_clipped[i, 1]),
                y_min_clipped=float(y_clipped[i, 0]),
                y_max_clipped=float(y_clipped[i, 1]),
            )
            output.append(oe)

        return output

    # _evaluate_k можно удалить (оставлен для совместимости, но уже не используется)
    def _evaluate_k(self, k: int, df_avg: float):
        raise NotImplementedError("_evaluate_k устарел — используйте evaluate_k_worker и draw_echelegramma.")

    # def _evaluate_k(
    #         self, k: int, df_avg: float
    # ) -> [float, float, float, float]:
    #     """
    #     Рассчет положения x и y для краев порядка с учетом ограничения детектором
    #     :param k: порядок
    #     :return: ([float, float], [float, float]) положения x и y для краев порядка с учетом ограничения детектором
    #     """
    #
    #     x_min, y_min, x_max, y_max, _, _ = find_order_edges(k, self.prism_wedge_angle_rad,
    #                                                         self.prism_incidence_angle_rad,
    #                                                         self.grating_groove_tilt_rad, self.focal,
    #                                                         self.lines_in_mm, self.gamma_rad,
    #                                                         df_avg, self.glass_type)
    #
    #     xs = np.array([x_min, x_max])
    #     ys = np.array([y_min, y_max])
    #     # ys = ys - ys.max()
    #
    #     #x_clipped = np.clip(xs, -self.matrix_size / 2 + self.dx, self.matrix_size / 2 + self.dx)
    #     y_clipped = np.clip(ys, 0, self.matrix_size / 2 * 2 + self.dy)
    #
    #     return [x_min, x_max, y_min, y_max]
