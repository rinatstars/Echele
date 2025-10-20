from __future__ import annotations
import math
import os
import logging
import ctypes
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, get_context, Value
from lib.echelle.dataClasses import (
    Grating, Prism, Spectrometer,
    EvaluationResult, ConfigOGF
)
from lib.echelle.echelleMath import (
    lambda_range, diffraction_angle,
    angle_diffraction_prism,
    lambda_from_diffraction_angle,
    find_orders_range, find_order_edges,
    prism_incidence_angle_rad,
    grating_groove_tilt_rad,
    lambda_center_k
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# === Константы ===

LAMBDA_MIN = 167  # нм
LAMBDA_MAX = 780  # нм
LAMBDA_CTR = 200  # нм

# глобальная переменная для больших неизменяемых структур (инициализируется через initializer)
_SPECTRA_LINES: Optional[np.ndarray] = None

def init_worker(spectra_lines: Optional[Sequence] = None) -> None:
    """
    Initializer for worker processes: store large read-only structures in module-global variable.
    Called once per worker process (via Pool initializer).
    """
    global _SPECTRA_LINES
    if spectra_lines is None:
        _SPECTRA_LINES = None
    else:
        # Convert to numpy array for efficient access in workers
        try:
            _SPECTRA_LINES = np.asarray(spectra_lines)
        except Exception:
            # Fallback: keep as-is
            _SPECTRA_LINES = spectra_lines


# -----------------------
# Функции верхнего уровня для многопоточности
# -----------------------
def calc_focal_for_resolution(lines_in_mm: float, gamma_rad: float, k: int, slit_width: float, res_limit: float) -> float:
    """Ищем f (мм), чтобы обеспечить заданное разрешение."""
    dispers = slit_width / res_limit  # um->mm? keep as original formula style
    return 1000.0 * ((dispers * math.cos(gamma_rad)) / (k * lines_in_mm))


def clipped_spectral_loss_for_order(
    k: int,
    f: float,
    lines_in_mm: float,
    gamma_rad: float,
    grating_cross_tilt_rad: float,
    prism_rad: float,
    a: float,
    phi2: float,
    df_avg: float,
    df_prism_min: float,
    matrix_size: float,
    glass_type: str,
) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Для порядка k:
    - строим спектральную линию ±10% от λ_center,
    - вырезаем по X=±LIMIT,
    - возвращаем (lam_full, lam_vis, lam_lost, lam_lost_left, lam_lost_right)
    """
    # find_order_edges returns x_min, y_min, x_max, y_max, lambdamin, lambdamax
    x_min, _, x_max, _, lammin, lammax = find_order_edges(
        k, prism_rad, a, phi2, f, lines_in_mm, gamma_rad, grating_cross_tilt_rad,
        df_avg, df_prism_min, glass_type
    )

    # if any None, propagate None (caller should handle)
    if x_min is None or x_max is None or lammin is None or lammax is None:
        return 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0)

    xs = np.array([x_min, x_max], dtype=float)
    lams = np.array([lammin, lammax], dtype=float)
    lam_full = float(lams.max() - lams.min())

    # обрезаем по X
    half = float(matrix_size) / 2.0
    xs_clip = np.clip(xs, -half, half)
    df_clip = df_avg - np.arcsin(xs_clip / float(f))
    lam_clip = lambda_from_diffraction_angle(k, df_clip, lines_in_mm, gamma_rad + grating_cross_tilt_rad)

    lam_vis = float(np.nanmax(lam_clip) - np.nanmin(lam_clip))
    lam_lost_left = (float(lams.min()), float(np.nanmin(lam_clip)))
    lam_lost_right = (float(np.nanmax(lam_clip)), float(lams.max()))
    lam_lost = float(lam_full - lam_vis)
    return lam_full, lam_vis, lam_lost, lam_lost_left, lam_lost_right, lam_clip


def compute_spectral_losses(
    kmin: int,
    kmax: int,
    f: float,
    lines_in_mm: float,
    gamma_rad: float,
    grating_cross_tilt_rad: float,
    prism_rad: float,
    a: float,
    phi2: float,
    df_avg: float,
    df_prism_min: float,
    matrix_size: float,
    glass_type: str,
    max_lost_line: Optional[int] = None,
) -> Optional[
    tuple[float, float, float, int, list[tuple[Any, ...]], list[tuple[Any, ...]], dict[int, tuple[float, float]]]]:
    """
    Compute totals for spectral coverage across orders [kmin..kmax].
    Uses _SPECTRA_LINES global (initialized in worker via init_worker).
    Returns None if constraint (max_lost_line) exceeded or on error.
    """
    global _SPECTRA_LINES

    total_full = total_vis = total_lost = 0.0
    total_lost_lines = 0
    lost_line_list: List[Tuple[Any, ...]] = []
    vis_line_list: List[Tuple[Any, ...]] = []

    orders_in_lambda = {}

    for k in range(kmin, kmax + 1):
        lam_f, lam_v, lam_l, lam_l_left, lam_l_right, lam_clip = clipped_spectral_loss_for_order(
            k, f, lines_in_mm, gamma_rad, grating_cross_tilt_rad,  prism_rad, a, phi2, df_avg, df_prism_min, matrix_size, glass_type
        )

        orders_in_lambda[k] = lam_clip

        # Convert lam limits to nm for comparison with typical spectral line lists (which are usually in nm)
        lam_l_left_nm = np.array(lam_l_left) * 1e6
        lam_l_right_nm = np.array(lam_l_right) * 1e6

        # Accumulate
        total_full += lam_f
        total_vis += lam_v
        total_lost += lam_l

        # If there are spectral lines, check visibility/lost
        # _SPECTRA_LINES assumed shape (N, M) where second column index 1 contains wavelength in nm
        if _SPECTRA_LINES is not None:
            try:
                for row in _SPECTRA_LINES:
                    # row[1] is wavelength in nm (as in your code)
                    wl_nm = row[1]
                    if (lam_l_left_nm[0] < wl_nm < lam_l_left_nm[1]) or (lam_l_right_nm[0] < wl_nm < lam_l_right_nm[1]):
                        total_lost_lines += 1
                        lost_line_list.append(tuple(row))
                    elif lam_l_right_nm[0] > wl_nm > lam_l_left_nm[1]:
                        vis_line_list.append(tuple(row))
            except Exception:
                # If spectra lines format unexpected -> skip marking lines but still return totals
                logger.debug("spectra lines processing failed in worker", exc_info=True)

    return total_full, total_vis, total_lost, total_lost_lines, vis_line_list, lost_line_list, orders_in_lambda


def evaluate_grating(task: Tuple[float, float, Dict[str, Any]]) -> Optional[Spectrometer]:
    """
    Worker-callable function. Receives a task tuple:
      (line_in_mm: float, gamma_rad: float, cfg: dict)
    cfg must contain only JSON-serializable primitives (numbers, strings).
    """

    try:
        line_in_mm, gamma_rad, cfg = task
        # Unpack config
        matrix_size = float(cfg["matrix_size"])
        glass_type = str(cfg["glass_type"])
        lambda_min = float(cfg["lambda_min"])
        lambda_max = float(cfg["lambda_max"])
        lambda_ctr = float(cfg["lambda_ctr"])
        slit_width = float(cfg["slit_width"])
        res_limit = float(cfg["res_limit"])
        max_focal = float(cfg["max_focal"])
        max_angle = float(cfg.get("max_angle", 60))
        alfa = float(cfg.get("alfa", 0))
        betta = float(cfg.get("betta", -7))
        min_dist_k = float(cfg["min_dist_k"])
        max_focal_matrix_ratio = cfg.get("max_focal_matrix_ratio")
        max_lost_line = cfg.get("max_lost_line", 0)
        grating_cross_tilt_rad = cfg.get("grating_cross_tilt_rad", 0)

        # 1) диапазон порядков
        kmin, kmax = find_orders_range(line_in_mm, gamma_rad, lambda_min, lambda_max)
        if kmin == 0 or kmax == 0:
            return None

        # 2) фокусное расстояние
        _, k_ctr = find_orders_range(line_in_mm, gamma_rad, lambda_ctr, lambda_max)
        if k_ctr == 0:
            return None
        f = calc_focal_for_resolution(line_in_mm, gamma_rad, k_ctr, slit_width, res_limit)
        if f is None:
            return None
        # Check ratio limits if provided
        if max_focal_matrix_ratio is not None and (f / matrix_size) < float(max_focal_matrix_ratio):
            return None
        if f > float(max_focal):
            return None

        # 3) оптимальный угол призмы (iterate like original; local copy of logic)
        # We'll reuse the approach from original `_find_best_prism_angle` but keep it local here.
        best: Optional[Tuple[float, float, float]] = None
        prism_alfa = 10.0
        lambda1 = lambda_center_k(line_in_mm, gamma_rad, 1) * 1e-6

        # рассчет средней длины волны и дифракции для максимального порядка для сдвига Эшеллеграммы
        lambdamin, lambdamax = lambda_range(lambda1, kmax)
        if lambdamin is None or lambdamax is None:
            return None
        lambda_avg = (lambdamin + lambdamax) / 2
        df_avg = diffraction_angle(kmax, lambda_avg, line_in_mm, gamma_rad + grating_cross_tilt_rad)

        df_prism_min_optimal = 0.0

        while prism_alfa <= max_angle:
            prism = math.radians(prism_alfa)
            a = prism_incidence_angle_rad(alfa, prism)          # угол падения на призму
            phi2 = grating_groove_tilt_rad(prism, alfa, betta)  # угол наклона решетки вдоль штриха
            df_prism_min = angle_diffraction_prism(lambdamin, prism, glass_type, a, phi2)

            # границы спектра по X
            # находим расстояние между крайними порядками
            try:
                _, y_min_kmin, _, y_max_kmin, _, _ = \
                    find_order_edges(kmin, prism, a, phi2, f, line_in_mm, gamma_rad, grating_cross_tilt_rad,
                                     df_avg, df_prism_min, glass_type)
                _, y_min_kmin_p1, _, y_max_kmin_p1, _, _ = \
                    find_order_edges(kmin + 1, prism, a, phi2, f, line_in_mm, gamma_rad, grating_cross_tilt_rad,
                                     df_avg, df_prism_min, glass_type)
                _, y_min_kmax, _, y_max_kmax, _, _ = \
                    find_order_edges(kmax, prism, a, phi2, f, line_in_mm, gamma_rad, grating_cross_tilt_rad,
                                     df_avg, df_prism_min, glass_type)
            except Exception:
                y_min_kmin = y_min_kmin_p1 = y_min_kmax = y_max_kmin = y_max_kmin_p1 = y_max_kmax = None

            if y_min_kmin is None or y_min_kmin_p1 is None or y_max_kmax is None:
                if best is not None:
                    break
                prism_alfa += 0.5
                continue

            # Смещение по Y: y_min выбранного порядка должен стать self.dy
            y_max_kmin = y_max_kmin - y_min_kmax
            y_min_kmin = y_min_kmin - y_min_kmax
            y_min_kmax = 0.0

            y_kmin_mean = (y_max_kmin + y_min_kmin) / 2
            y_kmin_p1_mean = (y_max_kmin_p1 + y_min_kmin_p1) / 2

            dist = abs(y_kmin_p1_mean - y_kmin_mean)        # расстояние между соседними минимальными порядками
            dist_min_max = abs(y_min_kmax - y_max_kmin)     # расстояние между минимальным и максимальным порядками

            if dist < min_dist_k:
                prism_alfa += 0.5
                continue

            if dist_min_max > matrix_size:
                break

            gap = (matrix_size - dist_min_max) / 2.0
            if best is None or gap < best[2]:
                best = (prism_alfa, phi2, gap)
                df_prism_min_optimal = df_prism_min

            prism_alfa += 0.5

        if best is None:
            return None

        prism_deg, phi2, gap = best
        prism_rad = math.radians(prism_deg)
        a = prism_incidence_angle_rad(0.0, prism_rad)

        # 4. считаем спектральные потери _SPECTRA_LINES (init_worker must have been called)
        spectral = compute_spectral_losses(
            kmin, kmax, f, line_in_mm, gamma_rad, grating_cross_tilt_rad,
            prism_rad, a, phi2, df_avg, df_prism_min_optimal,
            matrix_size, glass_type, max_lost_line
        )

        if spectral is None:
            spectral = 0, 0, 0, 0, 0, 0, 0

        total_full, total_vis, total_lost, total_lost_line, vis_line_list, lost_line_list, orders_in_lambda = spectral
        if max_lost_line is not None and max_lost_line != 0 and total_lost_line > max_lost_line:
            return None

        loss_pct = 100.0 * total_lost / total_full if total_full > 0 else 0.0

        # Compose dataclasses for result (units preserved as in original)
        grating = Grating(
            lines_per_mm=float(line_in_mm),
            gamma_rad=float(gamma_rad),
            gr_cross_tilt_rad=float(grating_cross_tilt_rad),
            grating_tilt_deg=-7,
        )
        prism = Prism(
            wedge_angle_rad=float(prism_rad),
            tilt_deg=0.0,
            glass_type=glass_type,
        )
        result = EvaluationResult(
            N=round(float(line_in_mm), 1),
            gamma_deg=math.degrees(float(gamma_rad)),
            kmin=int(kmin),
            kmax=int(kmax),
            f_mm=round(float(f), 1),
            prism_deg=float(prism_deg),
            gap_mm=round(float(gap), 3),
            ratio_f_det=round(float(f) / float(matrix_size), 1),
            spectral_full_nm=round(float(total_full) * 1e6, 1),
            spectral_visible_nm=round(float(total_vis) * 1e6, 1),
            spectral_lost_nm=round(float(total_lost) * 1e6, 1),
            num_lost_lines=int(total_lost_line),
            loss_pct=round(loss_pct),
            visible_lines=vis_line_list,
            lost_lines=lost_line_list,
        )
        spectrometr = Spectrometer(
            grating=grating,
            prism=prism,
            focal_mm=float(f),
            matrix_size_mm=float(matrix_size),
            result=result,
            orders_in_lambda=orders_in_lambda,
            df_avg=float(df_avg),
            df_prism_min=float(df_prism_min_optimal),
        )
        return spectrometr

    except Exception:
        logger.exception("Unhandled exception in worker for task=%r", task)
        return None


def load_df_from_excel(filename: str):
    """
    Загрузка датафрейма из Excel
    :param filename: путь к файлу
    :return: dataframe
    """
    try:
        if not os.path.exists(filename):
            logger.info("Файл не найден: %s", filename)
            return
        return pd.read_excel(filename)
    except Exception as e:
        logger.warning("Ошибка чтения файла: %s", e, exc_info=True)


class OptimalGratingFinder:
    def __init__(self, config: ConfigOGF):
        """
        :param config: конфигурационный датасет для OptimalGratingFinder
        """
        self.load_config(config)
        self.vis_los_lines = None
        self.grating_cross_tilt_rad = math.radians(config.grating_cross_tilt_deg)

        # runtime populated
        self.spectra_lines_list: Optional[np.ndarray] = None
        self.grating_list: Optional[np.ndarray] = None
        self.spectrometers: List[Spectrometer] = []
        self.optimal_grating_dataframe: Optional[pd.DataFrame] = None

    def load_spectra_lines_list_from_excel(self, filename: str) -> None:
        """
        Load spectral lines table from Excel. Expected format: array-like with wavelength in column 1 (nm).
        """
        try:
            df = load_df_from_excel(filename)
            if "element" not in df and "lambda" not in df:
                logger.info("Формат файла со списом линий не верный: %s", filename)
                raise Exception(
                    "Формат файла со списом линий не верный")
            else:
                self.spectra_lines_list = df.values
        except Exception as e:
            logger.warning("Ошибка чтения файла со списком спектральных линий: %s", e, exc_info=True)

    def load_grating_list_from_excel(self, filename: str) -> None:
        """
        Загрузка списка дифракционных решеток из Excel
        :param filename: путь к файлу со списком решеток
        :return:
        """
        try:
            df = load_df_from_excel(filename)
            if ("N" not in df and "size_x" not in df and
                    "size_y" not in df and "gamma" not in df):
                logger.info("Формат файла со списом дифракционных решеток не верный: %s", filename)
                raise Exception(
                    "Формат файла со списом дифракционных решеток не верный")
            else:
                self.grating_list = df.values
        except Exception as e:
            logger.warning("Ошибка чтения файла со списком спектральных линий: %s", e, exc_info=True)

    def load_config(self, config: ConfigOGF):
        self.config = config
        self.lines_in_mm = config.lines_in_mm
        self.gamma_rad = np.radians(config.gamma_deg)
        self.lambda_min = config.lambda_min
        self.lambda_max = config.lambda_max
        self.lambda_ctr = config.lambda_ctr
        self.matrix_size = config.matrix_size
        self.glass_type = config.glass_type
        self.slit_width = config.slit_width
        self.res_limit = config.res_limit
        self.max_focal = config.max_focal
        self.min_dist_k = config.min_dist_k
        self.max_focal_matrix_ratio = config.max_focal_matrix_ratio
        self.max_lost_line = config.max_lost_line

    def _build_worker_config(self) -> Dict[str, Any]:
        """
        Create a minimal serializable configuration dict for workers.
        """
        return {
            "matrix_size": float(self.matrix_size),
            "glass_type": str(self.glass_type),
            "lambda_min": float(self.lambda_min),
            "lambda_max": float(self.lambda_max),
            "lambda_ctr": float(self.lambda_ctr),
            "slit_width": float(self.slit_width),
            "res_limit": float(self.res_limit),
            "max_focal": float(self.max_focal),
            "max_angle": 60.0,
            "alfa": 0.0,
            "betta": -7.0,
            "min_dist_k": float(self.min_dist_k),
            "max_focal_matrix_ratio": self.max_focal_matrix_ratio,
            "max_lost_line": self.max_lost_line,
            "grating_cross_tilt_rad": self.grating_cross_tilt_rad
        }

    def search_optimal(
            self,
            save_excel: bool = False,
            use_grating_list: bool = False,
    ) -> Optional[Tuple[pd.DataFrame, Optional[Spectrometer], Optional[Spectrometer]]]:
        """Перебираем line_in_mm (1) и gamma (0.5°) в параллели.
        :param use_grating_list: bool=False использовать список дифракционных решеток
        :param save_excel: bool=False сохранить файл .xlsx
        :return: DataFrame список оптимальных решоток или None если ничего не найдено
        """

        line_in_mm_vals: np.ndarray = np.arange(self.lines_in_mm[0], self.lines_in_mm[1], 0.1)
        gamma_vals: np.ndarray = np.arange(self.gamma_rad[0], self.gamma_rad[1], np.radians(0.5))

        # assemble tasks: send gamma in radians to workers
        tasks: List[Tuple[float, float, Dict[str, Any]]] = []
        cfg = self._build_worker_config()
        if not use_grating_list:
            for line_n in line_in_mm_vals:
                for gamma_rad in gamma_vals:
                    tasks.append((float(line_n), gamma_rad, cfg))
        else:

            if self.grating_list is not None:
                grating_list = np.asarray(self.grating_list)
                for row in grating_list:
                    tasks.append((row[0], math.radians(row[3]), cfg))


        total = len(tasks)
        if total == 0:
            return None

        # 🔹 общий счётчик завершённых заданий
        done = Value(ctypes.c_int, 0)
        self.done = done
        self.total = total
        step = total/10
        batch = 0

        # prepare worker initializer args
        init_args = (self.spectra_lines_list,) if self.spectra_lines_list is not None else (None,)

        results: List[Spectrometer] = []

        # create pool with initializer (so large arrays get loaded once per worker)
        workers = max(1, min(total, cpu_count()))
        chunksize = max(1, total // (workers * 8))
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if workers > 1:
            init_worker(*init_args) if isinstance(init_args, tuple) else init_worker(init_args)
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=init_args) as executor:
                # executor.map возвращает итератор, не создавая тысячи Future сразу
                for res in executor.map(evaluate_grating, tasks, chunksize=chunksize):
                    # res может быть None или Spectrometer
                    if res is not None:
                        results.append(res)

                    batch += 1
                    if batch % step == 0 or self.done.value + batch >= total:
                        with self.done.get_lock():
                            self.done.value += batch
                        batch = 0
        else:
            init_worker(*init_args)
            for task in tasks:
                res = evaluate_grating(task)
                if res is not None:
                    results.append(res)

                batch += 1
                if batch % step == 0 or done.value + batch >= total:
                    with done.get_lock():
                        done.value += batch
                    batch = 0

        valid = [r for r in results if r is not None]

        if len(valid):
            df = pd.DataFrame([r.result.to_dict(drop_lists=True) for r in valid])
            df.sort_values(["num_lost_lines", "spectral_lost_nm", "gap_mm"], inplace=True)
            self.optimal_grating_dataframe = df
            valid = [valid[i] for i in df.index]
            self.spectrometers = valid
            if save_excel:
                self.save_optimal_grating_dataframe_to_excel(f"opt_grat_det({self.matrix_size / 2 * 2})_split"
                                                             f"({self.slit_width})_res({self.res_limit * 1000}).xlsx")
            return df

        return None

    def save_optimal_grating_dataframe_to_excel(self, file_name: str):
        """
        Сохранение списка оптимальных решеток в файл Excel

        :param file_name: имя файла excel
        :return: None если не задано имя файла
        """
        if file_name is None:
            return None

        df = pd.DataFrame([r.to_dict(drop_lists=True) for r in self.optimal_grating_dataframe])
        df.to_excel(
            file_name,
            index=False)
