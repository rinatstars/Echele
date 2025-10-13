
from graf import k1, p, disp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar

# Параметры задачи
angle_b = 61  # угол блеска в градусах
N = 150e3  # число штрихов на метр (75 на мм)
lambda_10 = (2 * np.sin(np.radians(angle_b))/N)  # длина волны блеска (в метрах)
nu_10 = 1 / lambda_10  # волновое число блеска (обратные метры)

# Диапазон длин волн (в метрах)
lambdas = np.linspace(240e-9, 350e-9, 50000)  # от 400 нм до 700 нм
nus = 1 / lambdas  # волновые числа (обратные метры)

# Максимальный коэффициент отражения
rho_m = 1.0  # нормировано на максимум

dl = 0.02 # 20 мкм
dlambda = 0.01 # 10 пм
f = 310
lambdaM = 380 # nm
l = 13 # mm

# Выражение для x_k
def x_k(k, nu, nu_10):
    return k - nu / nu_10

# Функция для вычисления f(x)
def f_x(k, nu, nu_10):
    x = x_k(k, nu, nu_10)
    if x == 0:  # избегаем деления на 0
        return 1.0
    return (np.sin(np.pi * x) / (np.pi * x))**2

k = 100

# Оптимизация: нахождение максимума f(x)
result = minimize_scalar(lambda nu: -f_x(k, nu, nu_10), bounds=(nu_10 * (k - 1), nu_10 * (k + 1)), method='bounded')

# Формула для коэффициента отражения
def reflection_coefficient(nu, nu_10, k=1):
    x = k - nu / nu_10
    return rho_m * (np.sin(np.pi * x) / (np.pi * x))**2 if x != 0 else 1.0

# Функции для нахождения пересечений
def intersection_k_minus_1(nu, nu_10, k):
    return reflection_coefficient(nu, nu_10, k) - reflection_coefficient(nu, nu_10, k - 1)

def intersection_k_plus_1(nu, nu_10, k):
    return reflection_coefficient(nu, nu_10, k) - reflection_coefficient(nu, nu_10, k + 1)

def find_intersection(nu_min, nu_max, nu_10, k, step=0.1):
    # Перебираем значения на интервале от nu_min до nu_max с заданным шагом
    nu_values = np.arange(nu_min, nu_max, step)
    for i in range(1, len(nu_values)):
        if intersection_k_plus_1(nu_values[i-1], nu_10, k) * intersection_k_plus_1(nu_values[i], nu_10, k) < 0:
            # Нашли изменение знака, используем линейную интерполяцию для нахождения точного пересечения
            nu_left = nu_values[i-1]
            nu_right = nu_values[i]
            # Линейная интерполяция для точного нахождения пересечения
            intersection = nu_left - intersection_k_plus_1(nu_left, nu_10, k) * (nu_right - nu_left) / \
                           (intersection_k_plus_1(nu_right, nu_10, k) - intersection_k_plus_1(nu_left, nu_10, k))
            return intersection
    return None  # если пересечения не найдено

# plt.figure()
# plt.plot(lambdas, [intersection_k_plus_1(nu, nu_10, 100) for nu in nus])
# plt.plot(lambdas, [intersection_k_minus_1(nu, nu_10, 100) for nu in nus])
# plt.show()
#
# plt.figure()
# plt.plot(nus, [intersection_k_plus_1(nu, nu_10, 100) for nu in nus])
# plt.plot(nus, [intersection_k_minus_1(nu, nu_10, 100) for nu in nus])
# plt.show()

# Сохраняем данные в Excel
data = []

# График для разных порядков
plt.figure(figsize=(12, 8))

k1_val = np.round(k1(lambdaM, disp(dl, dlambda), l), decimals=0).astype(np.int64)
kM = k1_val + 3* np.round(p(lambdaM, 160, k1_val)).astype(np.int64)

# Параметры для работы с порядками
for k in range(k1_val, kM):  # Диапазон порядков
    try:
        # Оптимизация: нахождение максимума f(x) для каждого порядка
        result = minimize_scalar(lambda nu: -f_x(k, nu, nu_10), bounds=(nu_10 * (k - 1), nu_10 * (k + 1)), method='bounded')
        nu_max = result.x  # Значение nu, при котором f(nu) максимизируется
        lambda_max = 1 / nu_max * 1e9  # Длина волны в нм
        if k == 100:
            print()

        # Пересечения с соседними порядками
        nu_left = brentq(intersection_k_plus_1, nu_10 * (k + 2 / 3), nu_10 * k, args=(nu_10, k))  # пересечение слева
        nu_right = brentq(intersection_k_minus_1, nu_10 * k, nu_10 * (k - 2 / 3), args=(nu_10, k))  # пересечение справа

        # Преобразование в длины волн
        lambda_left = 1 / nu_left * 1e9  # в нм
        lambda_right = 1 / nu_right * 1e9  # в нм

        # Сохраняем информацию о длине волны
        data.append([k, lambda_left, (lambda_max-lambda_left)/2+lambda_left, lambda_max, (lambda_right - lambda_max)/2+lambda_max, lambda_right])

        # Построение графика для каждого порядка
        nu_values = np.linspace(nu_left, nu_right, 100)  # Генерация значений nu для графика
        rho_values = [reflection_coefficient(nu, nu_10, k) for nu in nu_values]  # Коэффициент отражения
        lambdas = 1 / nu_values * 1e9  # Длины волн в нм
        plt.plot(lambdas, rho_values, label=f'{k}')
    except ValueError:
        pass  # Если не получилось найти пересечение, пропустим

# Сохранение данных в Excel
df = pd.DataFrame(data, columns=["Order", "Lambda_left (nm)", "Lambda_left_mid (nm)", "Lambda_max (nm)", "Lambda_right_mid (nm)", "Lambda_right (nm)"])
df.to_excel(f"reflection_coefficients_{N/1000}_{angle_b}.xlsx", index=False)

# Настройки графика
plt.title("Reflection Coefficient of Echelle Grating for Different Orders")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflection Coefficient")
plt.grid(True)
plt.legend()
plt.show()

print("Данные сохранены в файл reflection_coefficients.xlsx.")


print(f"Данные сохранены в файл reflection_coefficients_{N/1000}_{angle_b}.xlsx.")
