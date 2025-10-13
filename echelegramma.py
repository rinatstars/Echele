import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

grating_density_lpm = 85.8
blaze_angle_rad = math.radians(41)     # угол блеска
grating_cross_tilt_rad = math.radians(31.3)       # угол наклона решетки поперек штриха
kmin = 19                   # минимальный диф порядок
kmax = 92                   # максимальный диф порядок
k = kmax

focal_length_mm = 426                     # фокусное расстояние
slit_width_microns = 25                   # ширина щели

# Границы рабочей области (по 10.4 мм в каждую сторону)
LIMIT = 11.25
dx = 0  # сдвиг картины по x
dy = 0  # сдвиг картины по y

glass_id = 2  # Материал призмы: 1-BaF, 2-CaF, 3-SiO
prism_wedge_angle_rad = math.radians(13.5)     # угол клина призмы
prism_tilt_deg = 0                # угол наклона призмы из Земакса (сохранять знак)
prism_incidence_angle_rad = math.radians(-prism_tilt_deg +
                                         math.degrees(prism_wedge_angle_rad / 2))      # угол падения на призму
grating_tilt_deg = -7               # угол наклона решетки из Земакса (сохранять знак)
# угол наклона решетки вдоль штриха
grating_groove_tilt_rad = math.radians(
    90 - (math.degrees(prism_wedge_angle_rad / 2) - prism_tilt_deg - grating_tilt_deg))

I = []
lambd = []
z = []
lambda_avg = 0
df_avg = 0
spectral_range = []

lambda1 = 2 * np.sin(blaze_angle_rad) / grating_density_lpm
lambda2 = lambda1 * 0.9 / k
lambda3 = lambda1 * 1.1 / k


def refractive_index(lambda_nm, glass):
    """Функция вычисления показателя преломления для стекол BaF и CaF."""
    lambda_sq = lambda_nm ** 2
    if glass == 1:  # BaF
        n_sq = 1 + 0.33973 + \
               (0.81070 * lambda_sq) / (lambda_sq - 0.10065**2) + \
               (0.19652 * lambda_sq) / (lambda_sq - 29.87**2) + \
               (4.52469 * lambda_sq) / (lambda_sq - 53.82**2)
    elif glass == 2:  # CaF
        n_sq = 1 + 0.5675888 / (1 - (0.050263605 / lambda_nm) ** 2) + \
               0.4710914 / (1 - (0.1003909 / lambda_nm) ** 2) + \
               3.8484723 / (1 - (34.649040 / lambda_nm) ** 2)
    elif glass == 3:  # SiO
        n_sq = 1 + 0.6961663 * lambda_sq/(lambda_sq - 0.0684043 ** 2) + \
               0.4079426 * lambda_sq / (lambda_sq - 0.1162414 ** 2) + \
               0.8974794 * lambda_sq / (lambda_sq - 9.896161 ** 2)

    return np.sqrt(n_sq)


def diffraction_angle(k, lambd):
    """Вычисляет угол дифракции"""
    sin_theta = k * lambd * grating_density_lpm - math.sin(grating_cross_tilt_rad)
    if -1 <= sin_theta <= 1:
        return math.asin(sin_theta)
    return None  # Если выходит за пределы


def lambda_from_diffraction_angle(k, df):
    """Вычисляет длины волны по углу дифракции для массива углов df"""
    return (np.sin(df) + np.sin(grating_cross_tilt_rad)) / (k * grating_density_lpm)


while k > kmin - 1:
    # Перебор длин волн в заданном диапазоне
    lambda4 = np.arange(lambda2, lambda3, 0.000000001) # шаг 1 нм
    for lambdy in lambda4:
        F1 = np.sin(np.pi * (k - (lambda1 / lambdy)))
        F2 = np.pi * (k - (lambda1 / lambdy))
        F3 = (F1 / F2) ** 2 if F2 != 0 else 0
        if F3 > 0.405:
            I.append(F3)
            lambd.append(lambdy)

    if not lambd: # Если нет допустимых длин волн, пропускаем итерацию
        k -= 1
        lambda2 = lambda1 * 0.9 / k
        lambda3 = lambda1 * 1.1 / k
        continue  # Пропускаем итерацию, если нет допустимых значений


    lambdamax = max(lambd)
    lambdamin = min(lambd)
    # Определяем среднюю длину волны и средний угол дифракции
    if k == kmax:
        lambda_avg = (lambdamax + lambdamin) / 2
        df_avg = diffraction_angle(kmax, lambda_avg)
    x = []
    y = []

    array_ = np.arange(lambdamin, lambdamax, 6e-9)

    for lambdy in [lambdamin, lambdamax]:
        n = refractive_index(lambdy * 1000, glass_id)

        # Преломление луча на гранях призмы
        # первая грань
        prism_internal_angle_b = prism_wedge_angle_rad - math.asin(math.sin(prism_incidence_angle_rad) / n)
        # вторая грань
        prism_internal_angle_c = math.asin(math.sin(prism_internal_angle_b) * n)

        # отражение от призмы
        prism_exit_angle_c1 = math.pi - 2 * grating_groove_tilt_rad - prism_internal_angle_c

        # Обратное преломление от первой грани
        prism_exit_angle_d = prism_wedge_angle_rad - math.asin(math.sin(prism_exit_angle_c1) / n)
        # от второй грани
        sin_prism_exit = math.sin(prism_exit_angle_d) * n

        if not -1 <= sin_prism_exit <= 1:
            continue  # Пропускаем нефизические значения

        prism_exit_angle_e = math.asin(sin_prism_exit)

        l1 = focal_length_mm * math.sin(prism_exit_angle_e)
        z.append(l1)
        y.append(abs(l1 - max(z))+.2)

        df_value = diffraction_angle(k, lambdy)
        # Обрезка точек, выходящих за границы
        if df_value is None:
            continue

        df = df_avg - df_value
        l2 = focal_length_mm * math.sin(df)

        x.append(l2)

    if x and y:
        x = np.array(x)
        y = np.array(y)

        # Обрезка линий по габаритам детектора
        x_clipped = np.clip(x, -LIMIT+dx, LIMIT+dx)
        y_clipped = np.clip(y, 0, LIMIT*2+dy)

        plt.plot(x_clipped, y_clipped, color='b', alpha=1)
        plt.text(np.mean(x_clipped), np.mean(y_clipped), f'{k}', size=8, va='center' )

        df = df_avg - np.arcsin(x_clipped / focal_length_mm)

        lambdy_clipped = lambda_from_diffraction_angle(k, df)
        spectral_range.append([k, 1000 * lambdy_clipped[0],
                               1000 * (lambdy_clipped[0]+lambdy_clipped[1])/2, 1000 * lambdy_clipped[1]])

    k -= 1
    lambda2 = lambda1 * 0.9 / k
    lambda3 = lambda1 * 1.1 / k
    lambd.clear()


spectral_range = np.array(spectral_range)
df_spectral_range = pd.DataFrame(spectral_range, columns=['k','Start(um)', 'Middle(um)', 'End(um)'])
df_spectral_range.to_excel(f'spectral_range_{grating_density_lpm}_{math.degrees(blaze_angle_rad)}.xlsx', index=False, engine='openpyxl')

# Ограничение области графика
plt.xlim(-LIMIT+dx, LIMIT+dx)
plt.ylim(0, LIMIT*2+dy)
plt.xlabel("x (мм)")
plt.ylabel("y (мм)")
plt.title("Эшеллеграмма")
plt.show()
