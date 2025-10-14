from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Grating:
    lines_per_mm: float
    gamma_rad: float
    gr_cross_tilt_rad: float
    grating_tilt_deg: float  # если нужно


@dataclass(frozen=True)
class Prism:
    wedge_angle_rad: float
    tilt_deg: float
    glass_type: str  # "CaF" / "BaF"


@dataclass
class EvaluationResult:
    # числовые поля
    N: float
    gamma_deg: float
    kmin: int
    kmax: int
    f_mm: float
    prism_deg: float
    gap_mm: float
    ratio_f_det: float
    spectral_full_nm: float
    spectral_visible_nm: float
    spectral_lost_nm: float
    num_lost_lines: int
    loss_pct: float
    # списки линий
    visible_lines: List[Tuple[float, str]]  # [(wavelength_nm, label), ...]
    lost_lines: List[Tuple[float, str]]

    def to_dict(self, drop_lists: bool = True):
        d = {
            "N": round(self.N, 1),
            "gamma_deg": round(self.gamma_deg, 3),
            "kmin": self.kmin,
            "kmax": self.kmax,
            "f_mm": round(self.f_mm, 1),
            "prism_deg": round(self.prism_deg, 3),
            "gap_mm": round(self.gap_mm, 3),
            "ratio_f_det": round(self.ratio_f_det, 2),
            "spectral_full_nm": round(self.spectral_full_nm, 3),
            "spectral_visible_nm": round(self.spectral_visible_nm, 3),
            "spectral_lost_nm": round(self.spectral_lost_nm, 3),
            "num_lost_lines": self.num_lost_lines,
            "loss_pct": round(self.loss_pct, 1),
        }
        if not drop_lists:
            d["visible_lines"] = list(self.visible_lines)
            d["lost_lines"] = list(self.lost_lines)
        return d


@dataclass(frozen=True)
class Spectrometer:
    grating: Grating
    prism: Prism
    result: Optional[EvaluationResult]
    focal_mm: float
    matrix_size_mm: float
    df_avg: float = 0.0
    df_prism_min: float = 0.0

@dataclass(frozen=True)
class ConfigOGF:
    """
        Конфигурационные параметры оптимизатора эшелле-спектрометра (OptimalGratingFinder).

        Содержит диапазоны решёток, углов, диапазон длин волн и ограничения,
        используемые при расчёте и оптимизации спектральной системы.

        Атрибуты
        ---------
        lines_in_mm : [float, float]
            Диапазон частоты штрихов дифракционной решётки, штр/мм (например [100, 300]).
        gamma_deg : [float, float]
            Диапазон углов наклона решётки γ в градусах (например [0.0, 45.0]).
        lambda_min : float
            Минимальная длина волны спектра, мкм.
        lambda_max : float
            Максимальная длина волны спектра, мкм.
        lambda_ctr : float
            Центральная длина волны, мкм — используется для балансировки диапазона при расчёте эшеллеграммы.
        matrix_size : float
            Размер детектора по оси дисперсии, мм.
        glass_type : str
            Тип стекла призм (например 'BK7', 'SF11') — определяет дисперсию.
        slit_width : int
            Ширина входной щели, мкм.
        res_limit : float
            Минимально допустимое разрешение системы (безразмерно).
        max_focal : int
            Максимально допустимое фокусное расстояние объектива, мм.
        min_dist_k : float
            Минимально допустимое отношение расстояния от решётки до призмы.
        max_focal_matrix_ratio : int, optional
            Максимально допустимое отношение фокусного расстояния к размеру матрицы (по умолчанию None).
        max_lost_line : int, optional
            Максимально допустимое число потерянных спектральных линий (по умолчанию None).
        grating_cross_tilt_deg : float, default=6
            Поперечный наклон эшелле-решётки относительно входной щели, градусы.
        """
    lines_in_mm: [float, float]
    gamma_deg: [float, float]
    lambda_min: float
    lambda_max: float
    lambda_ctr: float
    matrix_size: float
    glass_type: str
    slit_width: int
    res_limit: float
    max_focal: int
    min_dist_k: float
    max_focal_matrix_ratio: int = None
    max_lost_line: int = None
    grating_cross_tilt_deg: float = 6
