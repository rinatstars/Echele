import logging
from lib.echelle.optimalGratingFinder import OptimalGratingFinder
from lib.gui.gui import EchelleGUI
from lib.echelle.zmx_echelle_editor import ZmxEchelleEditor
from lib.echelle.dataClasses import (
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
    configOGF = ConfigOGF([70, 120], [40, 80], LAMBDA_MIN, LAMBDA_MAX, LAMBDA_CTR,
                          LIMIT * 2, "CaF", SLIT_WIDTH, RES_LIMIT, MAX_FOCAL, MIN_DIST, MAX_RATIO, MAX_LOST_LINE)
    ogf = OptimalGratingFinder(configOGF)
    gui = EchelleGUI(ogf)
    gui.run()


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()