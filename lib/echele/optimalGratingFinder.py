from __future__ import annotations

import math
from typing import List, Optional

import pandas as pd
import numpy as np
import multiprocessing as mp
from  lib.echele.dataClasses import (
    Grating, Prism, Spectrometer,
    EvaluationResult
)
from lib.echele.echeleMath import (
    lambda_range, diffraction_angle,
    angle_diffraction_prism,
    lambda_from_diffraction_angle,
    find_orders_range, find_order_edges,
    prism_incidence_angle_rad,
    grating_groove_tilt_rad
)

# === Константы ===

LAMBDA_MIN = 167  # нм
LAMBDA_MAX = 780  # нм
LAMBDA_CTR = 200  # нм


class OptimalGratingFinder:
    def __init__(self, lines_in_mm: [float, float], gamma_deg: [float, float], lambda_min: float, lambda_max: float,
                 lambda_ctr: float, matrix_size: float, glass_type: str, slit_width: int,
                 res_limit: float, max_focal: int, min_dist_k: float, max_focal_matrix_ratio: int = None,
                 max_lost_line: int = None, grating_cross_tilt_deg: float = 6):
        """

        :param lines_in_mm: количество штрихов на миллиметр дифракционной решетки
        :param gamma_deg: угол блеска дифракционной решетки в градусах
        :param lambda_min: минимальная длина волны
        :param lambda_max: максимальная длина волны
        :param lambda_ctr: основная длина волны для которой оптимизировать разрешение
        :param matrix_size: размер детектора
        :param glass_type: материал призмы (CaF, BaF)
        :param slit_width: расстояние между спектральными линиями на детекторе для основной длины волны
        :param res_limit: разрешение в нм для центральной длины волны
        :param max_focal: максимальное фокусное расстояние до детектора
        :param min_dist_k: минимальное расстояние между минимальными порядками
        :param max_focal_matrix_ratio: максимальное отношение между фокусом и размером детектора
        :param max_lost_line: максимальная возможная потеря из списка спектральных линий. 0 - без ограничений
        """
        self.lines_in_mm = lines_in_mm
        self.gamma_rad = np.radians(gamma_deg)
        self.spectra_lines_list = None
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lamda_ctr = lambda_ctr
        self.matrix_size = matrix_size
        self.glass_type = glass_type
        self.slit_width = slit_width
        self.res_limit = res_limit
        self.max_focal = max_focal
        self.min_dist_k = min_dist_k
        self.max_focal_matrix_ratio = max_focal_matrix_ratio
        self.max_lost_line = max_lost_line
        self.vis_los_lines = None

        self.spectrometers: Optional[List[Spectrometer]] = None

    def load_spectra_lines_list_from_excel(self, filename: str) -> None:
        df = pd.read_excel(filename)

        self.spectra_lines_list = df.values

    def _calc_focal_for_resolution(self, lines_in_mm: float, gamma: float, k: int) -> float:
        """Ищем f (мм), чтобы обеспечить заданное разрешение."""
        dispers = self.slit_width / self.res_limit
        return 1000 * ((dispers * math.cos(gamma)) / (k * lines_in_mm))

    def _find_best_prism_angle(self, lines_in_mm: float, gamma: float, kmin: int, kmax: int, f: float,
                               max_angle: float = 60, alfa: float = 0, betta: float = -7):
        """
        Ищем prism_alfa ∈ [10°, 60°] шагом 0.5°,
        минимизируя зазор (gap) = LIMIT - (dist_min_max/2).

        :param lines_in_mm: количество штрихов на миллиметр дифракционной решетки
        :param gamma: угол блеска дифракционной решетки в градусах
        :param kmin: минимальный порядок дифракции
        :param kmax: максимальный порядок дифракции
        :param f: фокусное расстояние
        :param max_angle: максимальный угол призмы
        :param alfa: угол наклона призмы из Земакса
        :param betta: угол наклона решетки из Земакса
        :return:
        """

        best = None  # (prism_alfa, phi2, gap)

        prism_alfa = 10.0
        lambda1 = 2 * math.sin(gamma) / lines_in_mm  # в метрах

        # рассчет средней длины волны и дифракции для максимального порядка для сдвига Эшеллеграммы
        lambdamin, lambdamax = lambda_range(lambda1, kmax)
        lambda_avg = (lambdamax + lambdamin) / 2
        df_avg = diffraction_angle(kmax, lambda_avg, lines_in_mm, gamma)

        dist_min_max = 0
        dy = 0
        df_prism_min_optimal = 0

        while prism_alfa <= max_angle:
            prism = math.radians(prism_alfa)

            a = prism_incidence_angle_rad(alfa, prism)  # угол падения на призму
            phi2 = grating_groove_tilt_rad(prism, alfa, betta)  # угол наклона решетки вдоль штриха

            df_prism_min = angle_diffraction_prism(lambdamin, prism, self.glass_type, a, phi2)

            # границы спектра по X
            # находим расстояние между крайними порядками
            _, y_min_kmin, _, y_max_kmin, _, _ = find_order_edges(kmin, prism, a, phi2, f,
                                                                  lines_in_mm, gamma, df_avg, df_prism_min,
                                                                  self.glass_type)
            _, y_min_kmin_plus_1, _, y_max_kmin_plus_1, _, _ = find_order_edges(kmin + 1, prism, a, phi2, f,
                                                                                lines_in_mm, gamma, df_avg,
                                                                                df_prism_min, self.glass_type)
            _, y_min_kmax, _, y_max_kmax, _, _ = find_order_edges(kmax, prism, a, phi2, f,
                                                                  lines_in_mm, gamma, df_avg,
                                                                  df_prism_min, self.glass_type)
            if (not y_min_kmin_plus_1 or not y_min_kmin or not y_max_kmax
                    or not y_min_kmin):
                if best is not None:
                    return best, df_avg
                else:
                    break

            # 3) Смещение по Y: y_min выбранного порядка должен стать self.dy
            y_max_kmin = y_max_kmin - y_min_kmax
            y_min_kmin = y_min_kmin - y_min_kmax
            y_min_kmax = 0


            y_kmin_mean = (y_max_kmin + y_min_kmin) / 2
            y_kmin_plus_1_mean = (y_max_kmin_plus_1 + y_min_kmin_plus_1) / 2

            dist = abs(y_kmin_plus_1_mean - y_kmin_mean)  # расстояние между соседними минимальными порядками
            dist_min_max = abs(y_min_kmax - y_max_kmin)  # расстояние между минимальным и максимальным порядками

            if dist < self.min_dist_k:
                prism_alfa += 0.5
                continue

            if dist_min_max > self.matrix_size:  # 0.5 чтобы согластовать с изображением эшеллеграммы, была больше
                break

            gap = (self.matrix_size - dist_min_max) / 2
            if best is None or gap < best[2]:
                best = (prism_alfa, phi2, gap)
                df_prism_min_optimal = df_prism_min

            prism_alfa += 0.5

        return best, df_avg, df_prism_min_optimal

    def _clipped_spectral_loss_for_order(
            self,
            k: int,
            f: float,
            line_in_mm: float,
            gamma: float,
            prism: float,
            a: float,
            phi2: float,
            df_avg: float,
            df_prism_min: float
    ) -> tuple[float, float, float, [float, float], [float, float]]:
        """
        Для порядка k:
        - строим спектральную линию ±10% от λ_center,
        - вырезаем по X=±LIMIT,
        - возвращаем (λ_full, λ_visible, λ_lost).
        """

        x_min, _, x_max, _, lammin, lammax = find_order_edges(k, prism, a, phi2, f,
                                                              line_in_mm, gamma, df_avg,
                                                              df_prism_min, self.glass_type)

        xs = np.array([x_min, x_max])
        lams = np.array([lammin, lammax])
        lam_full = lams.max() - lams.min()

        # обрезаем по X
        xs_clip = np.clip(xs, -self.matrix_size / 2, self.matrix_size / 2)
        df_clip = df_avg - np.arcsin(xs_clip / f)
        lam_clip = lambda_from_diffraction_angle(k, df_clip, line_in_mm, gamma)

        lam_vis = lam_clip.max() - lam_clip.min()
        lam_lost_left = [lams.min(), lam_clip.min()]
        lam_lost_right = [lam_clip.max(), lams.max()]
        lam_lost = lam_full - lam_vis
        return lam_full, lam_vis, lam_lost, lam_lost_left, lam_lost_right

    def evaluate_grating(self, line_in_mm: float, gamma: float, ):
        df_avg = 0
        # 1. диапазон порядков
        kmin, kmax = find_orders_range(line_in_mm, gamma, self.lambda_min, self.lambda_max)
        if kmin == 0 or kmax == 0:
            return None

        # 2. фокусное расстояние
        f = self._get_focal_length(line_in_mm, gamma)
        if f is None:
            return None

        # 3. оптимальный угол призмы
        best, df_avg, df_prism_min = self._find_best_prism_angle(line_in_mm, gamma, kmin, kmax, f)
        if best is None:
            return None
        prism_deg, phi2, gap = best
        prism_rad = math.radians(prism_deg)
        a = prism_incidence_angle_rad(0, prism_rad)

        # 4. считаем спектральные потери
        spectral_data = self._compute_spectral_losses(
            kmin, kmax, f, line_in_mm, gamma, prism_rad, a, phi2, df_avg, df_prism_min)

        if spectral_data is None:
            return None
        total_full, total_vis, total_lost, total_lost_line, vis_line_list, lost_line_list = spectral_data

        if total_lost_line > self.max_lost_line != 0:
            return None

        loss_pct = 100 * total_lost / total_full if total_full > 0 else 0.0

        grating = Grating(
            lines_per_mm=line_in_mm,
            gamma_rad=gamma,
            gr_cross_tilt_rad=gamma,
            grating_tilt_deg=-7,
        )
        prism = Prism(
            wedge_angle_rad=prism_rad,
            tilt_deg=0,
            glass_type=self.glass_type
        )
        result = EvaluationResult(
            N=round(line_in_mm, 1),
            gamma_deg=math.degrees(gamma),
            kmin=kmin,
            kmax=kmax,
            f_mm=round(f, 1),
            prism_deg=prism_deg,
            gap_mm=round(gap, 3),
            ratio_f_det=round(f / self.matrix_size, 1),
            spectral_full_nm=round(total_full * 1e6, 1),
            spectral_visible_nm=round(total_vis * 1e6, 1),
            spectral_lost_nm=round(total_lost * 1e6, 1),
            num_lost_lines=total_lost_line,
            loss_pct=round(loss_pct),
            # списки линий
            visible_lines=vis_line_list,
            lost_lines=lost_line_list
        )
        spectrometr = Spectrometer(
            grating=grating,
            prism=prism,
            result=result,
            focal_mm=f,
            matrix_size_mm=self.matrix_size,
            df_avg=df_avg,
            df_prism_min=df_prism_min
        )

        return spectrometr

    def search_optimal(self, save_excel: bool = False) -> pd.DataFrame | None:
        """Перебираем line_in_mm (1) и gamma (0.5°) в параллели.
        :param save_excel: bool=False сохранить файл .xlsx
        :return: DataFrame список оптимальных решоток или None если ничего не найдено
        """

        line_in_mm_vals = np.arange(self.lines_in_mm[0], self.lines_in_mm[1], 0.1)
        gamma_vals = np.arange(self.gamma_rad[0], self.gamma_rad[1], 0.5)

        tasks = [(float(l_i_m), float(g)) for l_i_m in line_in_mm_vals for g in gamma_vals]
        with mp.Pool(mp.cpu_count()) as pool:
            results: List[Spectrometer] = pool.starmap(self.evaluate_grating, tasks)

        valid = [r for r in results if r is not None]
        if len(valid):
            df = pd.DataFrame([r.result.to_dict(drop_lists=True) for r in valid])
            df.sort_values(["num_lost_lines", "spectral_lost_nm", "gap_mm"], inplace=True)
            valid = [valid[i] for i in df.index]
            self.spectrometers = valid
            self.vis_los_lines = valid[1], valid[2]
            if save_excel:
                self.save_optimal_grating_dataframe_to_excel(f"opt_grat_det({self.matrix_size / 2 * 2})_split"
                                                             f"({self.slit_width})_res({self.res_limit * 1000}).xlsx")
            return df, valid[1], valid[2]

        return None

    def save_optimal_grating_dataframe_to_excel(self, file_name: str):
        """
        Сохранение списка оптимальных решеток в файл Excel

        :param file_name: имя файла excel
        :return: None если не задано имя файла
        """
        if file_name is None:
            return None

        df = pd.DataFrame([r.to_dict(drop_lists=True) for r in self.optimal_grating_result_list])
        df.to_excel(
            file_name,
            index=False)

    def _get_focal_length(self, line_in_mm: float, gamma: float) -> float | None:
        _, k_ctr = find_orders_range(line_in_mm, gamma, self.lamda_ctr, self.lambda_max)
        if k_ctr == 0:
            return None
        f = self._calc_focal_for_resolution(line_in_mm, gamma, k_ctr)
        ratio = f / self.matrix_size
        if ratio < self.max_focal_matrix_ratio or f > self.max_focal:
            return None
        return f

    def _compute_spectral_losses(self, kmin, kmax, f, line_in_mm, gamma,
                                 prism_rad, a, phi2, df_avg, df_prism_min):
        if self.spectra_lines_list is None:
            return 0.0, 0.0, 0.0, 0

        total_full = total_vis = total_lost = 0.0
        total_lost_line = 0
        lost_line_list = []
        vis_line_list = []
        for k in range(kmin, kmax + 1):
            lam_f, lam_v, lam_l, lam_l_left, lam_l_right = self._clipped_spectral_loss_for_order(
                k, f, line_in_mm, gamma, prism_rad, a, phi2,
                df_avg, df_prism_min)
            lam_l_left = np.array(lam_l_left) * 1e6
            lam_l_right = np.array(lam_l_right) * 1e6
            if lam_l > 0:
                for line in self.spectra_lines_list:
                    if (lam_l_left[0] < line[1] < lam_l_left[1] or
                            lam_l_right[0] < line[1] < lam_l_right[1]):
                        total_lost_line += 1
                        lost_line_list.append(line)
                    elif lam_l_right[0] > line[1] > lam_l_left[1]:
                        vis_line_list.append(line)
            total_full += lam_f
            total_vis += lam_v
            total_lost += lam_l

        if total_lost_line > self.max_lost_line != 0:
            return None

        return total_full, total_vis, total_lost, total_lost_line, vis_line_list, lost_line_list
