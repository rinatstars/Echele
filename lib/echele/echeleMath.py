from __future__ import annotations
import numpy as np
import math
from functools import lru_cache
from typing import Optional, Tuple


def prism_incidence_angle_rad(prism_tilt_deg: float,
                              prism_wedge_angle_rad: float) -> float:
    """
    Вычисление угола падения луча на первую грань призмы в радианах
    :param prism_tilt_deg: угол наклона призмы из Земакса (сохранять знак)
    :param prism_wedge_angle_rad: угол клина призмы
    :return: угол падения луча на первую грань призмы в радианах
    """
    return float(np.radians(-prism_tilt_deg + np.degrees(prism_wedge_angle_rad / 2)))


def grating_groove_tilt_rad(
        prism_wedge_angle_rad: float,
        prism_tilt_deg: float,
        grating_tilt_deg: float) -> float:
    """
    Вычисление угола наклона решетки вдоль штриха
    :param prism_wedge_angle_rad: угол клина призмы
    :param prism_tilt_deg: угол наклона призмы из Земакса (сохранять знак)
    :param grating_tilt_deg: угол наклона решетки из Земакса (сохранять знак)
    :return:
    """
    return float(np.radians(90 - (np.degrees(prism_wedge_angle_rad / 2) - prism_tilt_deg - grating_tilt_deg)))


@lru_cache(maxsize=4096)
def refractive_index(lambda_nm: float, glass: str = "CaF") -> float | None:
    """Schott‑подобная формула для CaF или BaF."""
    lam2 = lambda_nm ** 2
    if glass == "BaF":  # BaF
        n2 = (
                1
                + 0.33973
                + (0.81070 * lam2) / (lam2 - 0.10065 ** 2)
                + (0.19652 * lam2) / (lam2 - 29.87 ** 2)
                + (4.52469 * lam2) / (lam2 - 53.82 ** 2)
        )
    elif glass == "CaF":  # CaF
        n2 = (
                1
                + 0.5675888 / (1 - (0.050263605 / lambda_nm) ** 2)
                + 0.4710914 / (1 - (0.1003909 / lambda_nm) ** 2)
                + 3.8484723 / (1 - (34.649040 / lambda_nm) ** 2)
        )
    else:
        n2 = None
    return np.sqrt(n2)


@lru_cache(maxsize=4096)
def lambda_range(lambda1, k) -> tuple[float, float] | tuple[None, None]:
    """
    Поиск диапазона длин волн для порядка
    :param lambda1: длина волны первого порядка
    :param k: номер порядка
    :return:
    """
    lambda2 = lambda1 * 0.9 / k
    lambda3 = lambda1 * 1.1 / k

    # сразу создаем массив точек
    _lambda_range = np.linspace(lambda2, lambda3, 10000)

    # векторно считаем F2 и F1
    F2 = np.pi * (k - (lambda1 / _lambda_range))
    F1 = np.sin(F2)

    # безопасно считаем F3 (где F2!=0)
    F3 = np.zeros_like(F2)
    mask = F2 != 0
    F3[mask] = (F1[mask] / F2[mask]) ** 2

    # оставляем только точки, где F3 > 0.405
    mask_valid = F3 > 0.405
    lambda_valid = _lambda_range[mask_valid]

    if lambda_valid.size > 0:
        lambdamax = lambda_valid.max()
        lambdamin = lambda_valid.min()
        return lambdamin, lambdamax
    else:
        return None, None


@lru_cache(maxsize=4096)
def lambda_center_k(lines_in_mm: float, gamma: float, k: int) -> float:
    """Центральная длина волны (нм) для порядка k."""
    return (2 * math.sin(gamma) / lines_in_mm) * 1e6 / k


@lru_cache(maxsize=4096)
def find_orders_range(
        lines_in_mm: float, gamma: float, lam_min: float, lam_max: float
) -> tuple[int, int]:
    """Ищем kmin и kmax для диапазона длин волн."""
    kmin = 0
    kmax = 0
    lambda_0 = lambda_center_k(lines_in_mm, gamma, 1)
    k = 2
    while lambda_0 > lam_min:
        lambda_0 = lambda_center_k(lines_in_mm, gamma, k)
        if lambda_0 > lam_max:
            kmin = k
        kmax = k
        k += 1
        if k > 1000: break
    return kmin, kmax


@lru_cache(maxsize=4096)
def diffraction_angle(k: int, lam: float, lines_in_mm: float, gamma: float) -> float | None:
    """
    Угол дифракции решётки (радианы).

    :param k: порядок дифракции
    :param lam: длина волны
    :param lines_in_mm: количество штрихов на мм
    :param gamma: угол блеска
    """
    s = k * lam * lines_in_mm - math.sin(gamma)
    return (math.asin(s) - gamma) if -1 <= s <= 1 else None


def lambda_from_diffraction_angle(
        k: int, df: float, line_in_mm: float, phi: float
) -> np.ndarray:
    """Обратное уравнение решётки: df→λ."""
    return (np.sin(df + phi) + np.sin(phi)) / (k * line_in_mm)


@lru_cache(maxsize=4096)
def angle_diffraction_prism(lambdy: float, prism: float, glass_type: str,
                            phi: float, phi2: float = None):
    """
    Поиск угла дифракции призмы

    :param lambdy: длина волны
    :param prism: угол призмы
    :param glass_type: тип стекла призмы ("CaF")
    :param phi: угол падения на призму
    :param phi2: угол наклона решетки вдоль штриха
    :return: угол дифракции света после призмы
    """
    n = refractive_index(lambdy * 1000, glass_type)
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


@lru_cache(maxsize=4096)
def find_order_edges(
        k: int, prism: float,
        phi: float, phi2: float, f: float, lines_in_mm: float,
        gamma:  float, df_avg: float, df_prism_min: float,
        glass_type: str
) -> ([float, float, float, float, float, float] |
      [None, None, None, None, None, None]):
    """
    Поиск границ порядкка дифракции

    :param k: порядок дифракции
    :param prism: угол призмы
    :param phi: угол падения на призму
    :param phi2: угол наклона решетки вдоль штриха ???
    :param f: фокус
    :param lines_in_mm: количество штрихов на мм
    :param gamma: угол блеска
    :param df_avg: серединная дифракция для центровки
    :param df_prism_min: дифракция призмы для минимальной длины волны
    :param glass_type: тип стекла призмы ("CaF")
    :return: координаты Х и У, минимальная и максимальная длины волн
    """
    lambda1 = 2 * np.sin(gamma) / lines_in_mm

    lambdamin, lambdamax = lambda_range(lambda1, k)

    if not lambdamax or not lambdamin:
        return None, None, None, None, None, None  # Нет допустимых длин волн

    x, y = [], []
    for lambdy in [lambdamin, lambdamax]:

        e_value = angle_diffraction_prism(lambdy, prism, glass_type, phi, phi2)

        if not e_value:
            return None, None, None, None, None, None

        e = e_value - df_prism_min
        l1 = f * math.sin(e)
        y.append(l1)

        df_value = diffraction_angle(k, lambdy, lines_in_mm, gamma)
        if df_value is None:
            return None, None, None, None, None, None

        df = df_avg - df_value
        l2 = f * math.sin(df)
        x.append(l2)

    return x[0], y[0], x[1], y[1], lambdamin, lambdamax


def wavelength_to_detector_coords(
        lambda_nm: float,
        k_max: float,
        f_mm: float,
        lines_in_mm: float,
        gamma_rad: float,
        prism_tilt_deg: float,
        prism_wedge_angle_rad: float,
        df_avg: float,
        df_prism_min: float,
        glass_type: str,
        phi2: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
    """
    Вычислить координаты (x_mm, y_mm) на детекторе для заданной длины волны и порядка k.

    Возвращает (x_mm, y_mm) в миллиметрах, или None при отсутствии физического решения.

    Алгоритм повторяет логику find_order_edges, но для одиночной длины волны:
    1. вычисляем угол призмовой дифракции e = angle_diffraction_prism(lambda_mm, prism_rad, glass_type, phi, phi2)
       (phi — угол падения на призму; phi = prism_incidence_angle_rad(prism_tilt_deg, prism_wedge_angle_rad))
    2. y = f * sin(e - df_prism_min)  (с учётом сдвига df_prism_min, как в find_order_edges)
    3. df = diffraction_angle(k, lambda_mm, lines_in_mm, gamma_rad)
       x = f * sin(df_avg - df)   (как в find_order_edges: df = df_avg - df_value)

    :param lambda_nm: длина волны в нм
    :param k_max: максимальный порядок дифракции
    :param f_mm: фокус
    :param lines_in_mm:  количество штрихов на мм
    :param gamma_rad: угол блеска, рад
    :param prism_tilt_deg: угол наклона призмы из Земакс
    :param prism_wedge_angle_rad: угол клина призмы
    :param df_avg: серединная дифракция для центровки
    :param df_prism_min: дифракция призмы для минимальной длины волны
    :param glass_type: тип стекла призмы ("CaF")
    :param phi2: угол наклона решетки вдоль штриха
    :return: координаты x, y
    """

    # 1) prepare phi (angle of incidence on prism first face)
    lambda_mm = lambda_nm * 1e-6
    # in your earlier code: prism_incidence_angle_rad(prism_tilt_deg, prism_wedge_angle_rad)
    phi = prism_incidence_angle_rad(prism_tilt_deg, prism_wedge_angle_rad)

    k = k_max
    lambda1 = 2 * np.sin(gamma_rad) / lines_in_mm
    lambdamin, lambdamax = lambda_range(lambda1, k)

    while not(lambdamin < lambda_mm < lambdamax) and k != 1:
        k -= 1
        lambdamin, lambdamax = lambda_range(lambda1, k)

    # 3) prism diffraction angle e
    e_value = angle_diffraction_prism(lambda_mm, prism_wedge_angle_rad, glass_type, phi, phi2)
    if e_value is None:
        return None

    # In find_order_edges you subtracted df_prism_min: e = e_value - df_prism_min
    e = e_value - df_prism_min

    # guard domain for sin: sin accepts any real; but we'll compute y = f * sin(e)
    y_mm = -float(f_mm) * math.sin(e)

    # 4) diffraction angle for grating
    df_value = diffraction_angle(k, lambda_mm, lines_in_mm, gamma_rad)
    if df_value is None:
        return None

    df = df_avg - df_value
    x_mm = float(f_mm) * math.sin(df)

    return x_mm, y_mm
