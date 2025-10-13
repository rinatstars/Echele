# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import multiprocessing as mp
import pandas as pd
from functools import lru_cache

# === Константы ===
LIMIT = 10.4  # полуширина матрицы (мм)
GLASS_TYPE = 2  # 1 = BaF, 2 = CaF
SLIT_WIDTH = 21  # мкм
RES_LIMIT = 6.7e-3  # 4 пм = 0.004 нм
MAX_FOCAL = 800  # мм
MAX_RATIO = 10  # f / (2*LIMIT) < 15
MIN_DIST = 0.08  # расстояние между порядками
MAX_LOST_LINE = 0  # максимальная возможная потеря из списка спектральных линий. 0 - без ограничений

LAMBDA_MIN = 167  # нм
LAMBDA_MAX = 780  # нм
LAMBDA_CTR = 200  # нм


@lru_cache(maxsize=4096)
def refractive_index(lambda_nm: float, glass: int = GLASS_TYPE) -> float:
    """Schott‑подобная формула для CaF или BaF."""
    lam2 = lambda_nm ** 2
    if glass == 1:  # BaF
        n2 = (
                1
                + 0.33973
                + (0.81070 * lam2) / (lam2 - 0.10065 ** 2)
                + (0.19652 * lam2) / (lam2 - 29.87 ** 2)
                + (4.52469 * lam2) / (lam2 - 53.82 ** 2)
        )
    else:  # CaF
        n2 = (
                1
                + 0.5675888 / (1 - (0.050263605 / lambda_nm) ** 2)
                + 0.4710914 / (1 - (0.1003909 / lambda_nm) ** 2)
                + 3.8484723 / (1 - (34.649040 / lambda_nm) ** 2)
        )
    return np.sqrt(n2)


def lambda_center_k(gamma: float, N: float, k: int) -> float:
    """Центральная длина волны (нм) для порядка k."""
    return (2 * math.sin(gamma) / N) * 1e6 / k


def find_orders_range(
        N: float, gamma: float, lam_min: float, lam_max: float
) -> tuple[int, int]:
    """Ищем kmin и kmax для диапазона длин волн."""
    kmin = 0
    kmax = 0
    lambda_0 = lambda_center_k(gamma, N, 1)
    k = 2
    while lambda_0 > lam_min:
        lambda_0 = lambda_center_k(gamma, N, k)
        if lambda_0 > lam_max:
            kmin = k
        kmax = k
        k += 1
        if k > 1000: break
    return kmin, kmax


def diffraction_angle(
        k: int, lam: float, N: float, gamma: float
):
    """Угол дифракции решётки (радианы)."""
    s = k * lam * N - math.sin(gamma)
    return math.asin(s) if -1 <= s <= 1 else None


def calc_focal_for_resolution(
        N: float, gamma: float, res_limit: float, slit_mm: float, k: int
) -> float:
    """Ищем f (мм), чтобы обеспечить заданное разрешение."""
    dispers = slit_mm / res_limit
    return 1000 * ((dispers * math.cos(gamma)) / (k * N))


def lambda_from_diffraction_angle(
        k: int, df: float, N: float, phi: float
) -> np.ndarray:
    """Обратное уравнение решётки: df→λ."""
    return (np.sin(df) + np.sin(phi)) / (k * N)


def angle_diffraction_prism(lambdy: float, prism: float, phi: float,
                            glass: int = GLASS_TYPE, phi2: float = None):
    n = refractive_index(lambdy * 1000, glass)
    b = prism - math.asin(math.sin(phi) / n)
    if not -1 <= math.sin(b) * n <= 1:
        return None
    c = math.asin(math.sin(b) * n)
    if phi2 is None:
        return c
    c1 = math.pi - 2 * phi2 - c
    d = prism - math.asin(math.sin(c1) / n)

    sin_e = math.sin(d) * n
    if not -1 <= sin_e <= 1:
        return None

    e = math.asin(sin_e)

    return e


def clipped_spectral_loss_for_order(
        k: int,
        f: float,
        N: float,
        gamma: float,
        prism: float,
        a: float,
        phi2: float,
        df_avg: float,
) -> tuple[float, float, float, [float, float], [float, float]]:
    """
    Для порядка k:
    - строим спектральную линию ±10% от λ_center,
    - вырезаем по X=±LIMIT,
    - возвращаем (λ_full, λ_visible, λ_lost).
    """
    lambda1 = 2 * math.sin(gamma) / N  # в метрах

    x_min, y_min, x_max, y_max, lammin, lammax = find_order_edges(k, lambda1, prism, a, phi2, f, N, gamma, df_avg)

    xs = np.array([x_min, x_max])
    lams = np.array([lammin, lammax])
    lam_full = lams.max() - lams.min()

    # обрезаем по X
    xs_clip = np.clip(xs, -LIMIT, LIMIT)
    df_clip = df_avg - np.arcsin(xs_clip / f)
    lam_clip = lambda_from_diffraction_angle(k, df_clip, N, gamma)

    lam_vis = lam_clip.max() - lam_clip.min()
    lam_lost_left = [lams.min(), lam_clip.min()]
    lam_lost_right = [lam_clip.max(), lams.max()]
    lam_lost = lam_full - lam_vis
    return lam_full, lam_vis, lam_lost, lam_lost_left, lam_lost_right


def find_best_prism_angle(
        N: float, gamma: float, kmin: int, kmax: int, f: float, alfa: float = 0
):
    """
    Ищем prism_alfa ∈ [10°, 60°] шагом 0.5°,
    минимизируя gap = LIMIT - (dist_min_max/2).
    """
    best = None  # (prism_alfa, phi2, gap)
    dist = 0

    prism_alfa = 10.0
    lambda1 = 2 * math.sin(gamma) / N  # в метрах

    # рассчет средней длины волны и дифракции для максимального порядка для сдвига Эшеллеграммы
    lambdamin, lambdamax = lambda_range(lambda1, kmax)
    lambda_avg = (lambdamax + lambdamin) / 2
    df_avg = diffraction_angle(kmax, lambda_avg, N, gamma)

    while prism_alfa <= 60:
        prism = math.radians(prism_alfa)
        alfa = 0  # угол наклона призмы из Земакса
        betta = 6  # угол наклона решетки из Земакса
        a = math.radians(alfa + math.degrees(prism / 2))  # угол падения на призму

        phi2 = math.radians(90 - (math.degrees(prism / 2) - alfa + betta))

        # границы спектра по X
        # находим расстояние между крайними порядками
        x_min, y_min, crash1, crash2, _, _ = find_order_edges(kmin, lambda1, prism, a, phi2, f, N, gamma, df_avg)
        x_min2, y_min2, crash1, crash2, _, _ = find_order_edges(kmin + 1, lambda1, prism, a, phi2, f, N, gamma, df_avg)
        crash1, crash2, x_max, y_max, _, _ = find_order_edges(kmax, lambda1, prism, a, phi2, f, N, gamma, df_avg)
        if not y_min2 or not y_min or not y_max or not y_min or not x_min or not x_max:
            if best is not None:
                return best, df_avg
            else:
                break
        dist = abs(y_min2 - y_min)  # расстояние между соседними минимальными порядками
        dist_min_max = abs(y_max - y_min)  # расстояние между минимальным и максимальным порядками

        if dist < MIN_DIST:
            prism_alfa += 0.5
            continue

        if dist_min_max > 2 * LIMIT:
            break

        gap = LIMIT - dist_min_max / 2
        if best is None or gap < best[2]:
            best = (prism_alfa, phi2, gap)

        prism_alfa += 0.5

    return best, df_avg


def evaluate_grating(N: float, gamma: float, spectra_line_list: [str, float] = None, print_lost: bool = False):
    # 1. диапазон порядков
    kmin, kmax = find_orders_range(N, gamma, LAMBDA_MIN, LAMBDA_MAX)
    if kmin == 0 or kmax == 0:
        return None

    # 2. фокусное расстояние
    _, k_ctr = find_orders_range(N, gamma, LAMBDA_CTR, LAMBDA_MAX)
    if k_ctr == 0:
        return None
    f = calc_focal_for_resolution(N, gamma, RES_LIMIT, SLIT_WIDTH, k_ctr)
    ratio_f_det = f / (2 * LIMIT)
    if ratio_f_det < MAX_RATIO:
        return None
        #f = MAX_RATIO * 2 * LIMIT   # если фокус маленький, увеличиваем
                                    # до нужного соотношения с матрицей увеличивается разрешение
    if f > MAX_FOCAL:
        return None

    # 3. оптимальный угол призмы
    best, df_avg = find_best_prism_angle(N, gamma, kmin, kmax, f)
    if best is None:
        return None
    prism_deg, phi2, gap = best
    prism_rad = math.radians(prism_deg)
    a = math.radians(math.degrees(prism_rad) / 2)

    # 4. считаем спектральные потери
    total_full = total_vis = total_lost = 0.0
    total_lost_line = 0
    lost_line_list = []
    vis_line_list = []
    for k in range(kmin, kmax + 1):
        lam_f, lam_v, lam_l, lam_l_left, lam_l_right = clipped_spectral_loss_for_order(
            k, f, N, gamma, prism_rad, a, phi2,
            df_avg)
        lam_l_left = np.array(lam_l_left) * 1e6
        lam_l_right = np.array(lam_l_right) * 1e6
        if spectra_line_list is not None:
            if lam_l > 0:
                for line in spectra_line_list:
                    if (line[1] > lam_l_left[0] and line[1] < lam_l_left[1] or
                            line[1] > lam_l_right[0] and line[1] < lam_l_right[1]):
                        total_lost_line += 1
                        lost_line_list.append(line)
                    elif (line[1] > lam_l_left[1] and line[1] < lam_l_right[0]):
                        vis_line_list.append(line)
            total_full += lam_f
            total_vis += lam_v
            total_lost += lam_l


    if total_lost_line > MAX_LOST_LINE and MAX_LOST_LINE != 0:
        return None

    loss_pct = 100 * total_lost / total_full if total_full > 0 else 0.0

    if print_lost:
        print(lost_line_list)

    #print(f"{round(N, 1)}/{math.degrees(gamma)}: {kmin} - {kmax}, prism: {prism_deg}, loss: {total_lost}")

    return {
        "N": round(N, 1),
        "gamma_deg": math.degrees(gamma),
        "kmin": kmin,
        "kmax": kmax,
        "f_mm": f,
        "prism_deg": prism_deg,
        "gap_mm": gap,
        "ratio_f_det": ratio_f_det,
        "spectral_full_nm": total_full * 1e6,
        "spectral_visible_nm": total_vis * 1e6,
        "spectral_lost_nm": total_lost * 1e6,
        "num_lost_lines": total_lost_line,
        "loss_pct": round(loss_pct),
    }


def search_optimal() -> pd.DataFrame:
    """Перебираем N=20..150 (1), γ=40°..80° (0.5°) в параллели."""
    N_vals = np.arange(72, 125, 0.1)
    gamma_vals = np.radians(np.arange(40, 80.1, 0.5))

    df = pd.read_excel("spectral_line_list.xlsx")

    evaluate_grating(94.7, np.radians(45.8), df.values, print_lost=True)

    tasks = [(float(N), float(g), df.values) for N in N_vals for g in gamma_vals]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(evaluate_grating, tasks)

    valid = [r for r in results if r is not None]
    df = pd.DataFrame(valid)
    df.sort_values(["spectral_lost_nm", "gap_mm"], inplace=True)
    df.to_excel(f"opt_grat_det({LIMIT * 2})_split({SLIT_WIDTH})_res({RES_LIMIT * 1000}).xlsx", index=False)
    return df


if __name__ == "__main__":
    df_opt = search_optimal()
    print(df_opt.head(10).to_string(index=False))





def lambda_range(lambda1, k):
    lambda2 = lambda1 * 0.9 / k
    lambda3 = lambda1 * 1.1 / k

    # сразу создаем массив точек
    lambda_range = np.linspace(lambda2, lambda3, 400)

    # векторно считаем F2 и F1
    F2 = np.pi * (k - (lambda1 / lambda_range))
    F1 = np.sin(F2)

    # безопасно считаем F3 (где F2!=0)
    F3 = np.zeros_like(F2)
    mask = F2 != 0
    F3[mask] = (F1[mask] / F2[mask]) ** 2

    # оставляем только точки, где F3 > 0.405
    mask_valid = F3 > 0.405
    lambda_valid = lambda_range[mask_valid]

    if lambda_valid.size > 0:
        lambdamax = lambda_valid.max()
        lambdamin = lambda_valid.min()
        return lambdamin, lambdamax
    else:
        return None, None


def find_order_edges(k, lambda1, prism, phi, phi2, f, N, gamma, df_avg):
    lambdamin, lambdamax = lambda_range(lambda1, k)

    if not lambdamax or not lambdamin:
        return None, None, None, None  # Нет допустимых длин волн

    x, y = [], []
    for lambdy in [lambdamin, lambdamax]:

        e = angle_diffraction_prism(lambdy, prism, phi, GLASS_TYPE, phi2)

        if not e:
            return None, None, None, None, None, None

        l1 = f * math.sin(e)
        y.append(l1)

        df_value = diffraction_angle(k, lambdy, N, gamma)
        if df_value is None:
            return None, None, None, None, None, None

        df = df_avg - df_value
        l2 = f * math.sin(df)
        x.append(l2)

    return x[0], y[0], x[1], y[1], lambdamin, lambdamax
