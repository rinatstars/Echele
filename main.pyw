from lib.echele.optimalGratingFinder import OptimalGratingFinder
from lib.gui.gui import EcheleGUI
from lib.echele.dataClasses import (
    ConfigOGF
)



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


def main():
    configOGF = ConfigOGF([70, 120], [40,80], LAMBDA_MIN, LAMBDA_MAX, LAMBDA_CTR,
                               LIMIT*2, "CaF", SLIT_WIDTH, RES_LIMIT, MAX_FOCAL, MIN_DIST, MAX_RATIO, MAX_LOST_LINE)
    ogf = OptimalGratingFinder(configOGF)
    ogf.load_spectra_lines_list_from_excel("spectral_line_list.xlsx")
    gui = EcheleGUI(ogf)
    gui.run()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()


# lines_list = [
#     {"Element": "Ar", "lambda": 451.0733},
#     {"Element": "Ar", "lambda": 459.61},
#     {"Element": "Ar", "lambda": 462.8441},
#     {"Element": "Ar", "lambda": 487.6261},
#     {"Element": "Ar", "lambda": 488.7947},
#     {"Element": "Ar", "lambda": 522.1271},
#     {"Element": "Ar", "lambda": 525.276},
#     {"Element": "Ar", "lambda": 545.1652},
#     {"Element": "Ar", "lambda": 550.6113},
#     {"Element": "Ar", "lambda": 583.427},
#     {"Element": "Ar", "lambda": 589.98},
#     {"Element": "Ar", "lambda": 591.209},
#     {"Element": "Ar", "lambda": 592.8813},
#     {"Element": "Ar", "lambda": 594.267},
#     {"Element": "Ar", "lambda": 610.564},
#     {"Element": "Ar", "lambda": 617.018},
#     {"Element": "Ar", "lambda": 617.31},
#     {"Element": "Ar", "lambda": 662.247},
#     {"Element": "Ar", "lambda": 638.472},
#     {"Element": "Y", "lambda": 485.486},
#     {"Element": "Y", "lambda": 488.366},
#     {"Element": "Y", "lambda": 490.012},
#     {"Element": "Y", "lambda": 508.742},
#     {"Element": "Y", "lambda": 511.911},
#     {"Element": "Y", "lambda": 566.293},
#     {"Element": "H", "lambda": 486.129},
#     {"Element": "H", "lambda": 656.2852},
#     {"Element": "Sr", "lambda": 460.733},
#     {"Element": "Sr", "lambda": 548.086},
#     {"Element": "Sr", "lambda": 640.846},
#     {"Element": "Na", "lambda": 566.348},
#     {"Element": "Na", "lambda": 588.995},
#     {"Element": "Li", "lambda": 585.963},
#     {"Element": "Li", "lambda": 610.365},
#     {"Element": "Li", "lambda": 641.71},
#     {"Element": "Li", "lambda": 670.776},
#     {"Element": "Li", "lambda": 702.718},
#     {"Element": "W", "lambda": 638.472},
#     {"Element": "K", "lambda": 766.49},
#     {"Element": "K", "lambda": 769.896},
#     {"Element": "Rb", "lambda": 780.026},
#     {"Element": "Cs", "lambda": 662.801},
#     {"Element": "Cs", "lambda": 894.347},
#     {"Element": "Cs", "lambda": 852.113},
#     {"Element": "Ba", "lambda": 455.4033},
#     {"Element": "Ba", "lambda": 493.4077},
#     {"Element": "Ba", "lambda": 585.3675},
#     {"Element": "Hg", "lambda": 567.7105},
#     {"Element": "Hg", "lambda": 542.5253},
#     {"Element": "Hg", "lambda": 588.8939},
#     {"Element": "Hg", "lambda": 794.4555},
#     {"Element": "Ca", "lambda": 445.478},
#     {"Element": "Ca", "lambda": 643.907},
#     {"Element": "Ca", "lambda": 646.257},
#     {"Element": "Ca", "lambda": 811.898},
#     {"Element": "Cd", "lambda": 508.5822},
#     ]