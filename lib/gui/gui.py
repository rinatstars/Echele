"""Модуль графического интерфейса"""
import io
import math
import os
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import threading
import tkinter as tk
from tkinter import ttk, Tk, BooleanVar, StringVar, IntVar, Menu, DoubleVar, messagebox, filedialog
from tkintertable import TableCanvas, TableModel
from lib.echelle.optimalGratingFinder import OptimalGratingFinder
from lib.echelle.echellegrammaDrawer import EchellegrammaDrawer
from lib.echelle.zmx_echelle_editor import ZmxEchelleEditor
from lib.echelle.echelleMath import (
    wavelength_to_detector_coords,
    grating_groove_tilt_rad
)
from lib.echelle.dataClasses import ConfigOGF

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SETTINGS_FILENAME = os.path.join(".echelle_settings.json")


class EchelleGUI:
    """Класс графического интерфейса для управления устройством"""

    def __init__(self, ogfinder: OptimalGratingFinder):
        self.ogfinder = ogfinder
        self.config: ConfigOGF = self.ogfinder.config
        self.window = Tk()
        self._setup_window()
        self._init_variables()
        # Загружаем сохранённые настройки (если есть) — применяет значения к переменным
        self._load_settings()
        self._setup_ui()

    def _setup_window(self):
        """Настройка основного окна"""
        self.window.title("Эшелле решетки")
        self.window.geometry("1550x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_variables(self):
        """Инициализация переменных интерфейса"""
        self.last_spectral_line_file = "spectral_line_list.xlsx"
        self.last_grating_list_file = "grating_list.xlsx"
        self.lines_in_mm = [IntVar(value=80), IntVar(value=120)]
        self.gamma = [IntVar(value=40), IntVar(value=80)]
        self.limit = DoubleVar(value=10.4)  # полуширина матрицы (мм)
        self.glass_type = StringVar(value="CaF")  # 1 = BaF, 2 = CaF
        self.slit_width = IntVar(value=21)  # мкм
        self.res_limit = DoubleVar(value=6.7e-3)  # 4 пм = 0.004 нм
        self.max_focal = IntVar(value=800)  # мм
        self.max_ratio = IntVar(value=10)  # f / (2*LIMIT) < 15
        self.min_dist = DoubleVar(value=0.08)  # расстояние между порядками
        self.max_lost_line = IntVar(
            value=0)  # максимальная возможная потеря из списка спектральных линий. 0 - без ограничений

        self.lambda_min = IntVar(value=167)  # нм
        self.lambda_max = IntVar(value=780)  # нм
        self.lambda_ctr = IntVar(value=200)  # нм

        self.use_grating_list = BooleanVar(value=False)  # использовать список дифракционных решеток

        self.draw_line = BooleanVar(value=True)  # выводить список спектральных линий на эшелеграмме
        self.draw_line_lambda = BooleanVar(value=True)  # подписывать длины волн

        self.optimal_grating_dataframe = None
        self.active_spectrometer = None
        self.echellegramma_orders = None

        self.progress = None

    def _load_settings(self, path: str = SETTINGS_FILENAME):
        """
        Загружает настройки из JSON и применяет их к tk.Variable.
        Если файла нет или он битый — ничего не меняем.
        """
        try:
            if not os.path.exists(path):
                logger.debug("Settings file not found: %s", path)
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Не удалось загрузить настройки %s: %s", path, exc, exc_info=True)
            return

        try:
            # Простые числовые/строковые поля
            if "limit" in data:
                self.limit.set(data["limit"])
            if "glass_type" in data:
                self.glass_type.set(data["glass_type"])
            if "slit_width" in data:
                self.slit_width.set(data["slit_width"])
            if "res_limit" in data:
                self.res_limit.set(data["res_limit"])
            if "max_focal" in data:
                self.max_focal.set(data["max_focal"])
            if "max_ratio" in data:
                self.max_ratio.set(data["max_ratio"])
            if "min_dist" in data:
                self.min_dist.set(data["min_dist"])
            if "max_lost_line" in data:
                self.max_lost_line.set(data["max_lost_line"])

            if "lambda_min" in data:
                self.lambda_min.set(data["lambda_min"])
            if "lambda_max" in data:
                self.lambda_max.set(data["lambda_max"])
            if "lambda_ctr" in data:
                self.lambda_ctr.set(data["lambda_ctr"])

            # Списковые поля: lines_in_mm и gamma
            if "lines_in_mm" in data and isinstance(data["lines_in_mm"], (list, tuple)) and len(
                    data["lines_in_mm"]) >= 2:
                self.lines_in_mm[0].set(int(data["lines_in_mm"][0]))
                self.lines_in_mm[1].set(int(data["lines_in_mm"][1]))

            if "gamma" in data and isinstance(data["gamma"], (list, tuple)) and len(data["gamma"]) >= 2:
                self.gamma[0].set(int(data["gamma"][0]))
                self.gamma[1].set(int(data["gamma"][1]))

            # Булевы переключатели
            if "draw_line" in data:
                self.draw_line.set(bool(data["draw_line"]))
            if "draw_line_lambda" in data:
                self.draw_line_lambda.set(bool(data["draw_line_lambda"]))
            if "use_grating_list" in data:
                self.use_grating_list.set(bool(data["use_grating_list"]))

            # Восстановление геометрии окна (опционально)
            if "window_geometry" in data:
                try:
                    self.window.geometry(data["window_geometry"])
                except Exception:
                    pass

            # Загрузка файла с линиями используемыми последний раз
            if "last_spectral_line_file" in data:
                try:
                    self.last_spectral_line_file = data["last_spectral_line_file"]
                except Exception:
                    pass
            try:
                self.ogfinder.load_spectra_lines_list_from_excel(self.last_spectral_line_file)
            except Exception as e:
                logger.warning("Ошибка чтения файла со списком спектральных линий: %s", e, exc_info=True)

            # Загрузка файла со списком решеток используемыми последний раз
            if "last_grating_list_file" in data:
                try:
                    self.last_grating_list_file = data["last_grating_list_file"]
                except Exception:
                    pass
            try:
                self.ogfinder.load_grating_list_from_excel(self.last_grating_list_file)
            except Exception as e:
                logger.warning("Ошибка чтения файла со списком дифракционных решеток: %s", e, exc_info=True)

            logger.info("Loaded settings from %s", path)
        except Exception as exc:
            logger.warning("Ошибка применения настроек: %s", exc, exc_info=True)

    def _save_settings(self, path: str = SETTINGS_FILENAME):
        """
        Сохраняет текущие значения интерфейса в JSON-файл.
        """
        try:
            settings = {
                "lines_in_mm": [int(self.lines_in_mm[0].get()), int(self.lines_in_mm[1].get())],
                "gamma": [int(self.gamma[0].get()), int(self.gamma[1].get())],
                "limit": float(self.limit.get()),
                "glass_type": str(self.glass_type.get()),
                "slit_width": int(self.slit_width.get()),
                "res_limit": float(self.res_limit.get()),
                "max_focal": int(self.max_focal.get()),
                "max_ratio": int(self.max_ratio.get()),
                "min_dist": float(self.min_dist.get()),
                "max_lost_line": int(self.max_lost_line.get()),
                "lambda_min": int(self.lambda_min.get()),
                "lambda_max": int(self.lambda_max.get()),
                "lambda_ctr": int(self.lambda_ctr.get()),
                "draw_line": bool(self.draw_line.get()),
                "draw_line_lambda": bool(self.draw_line_lambda.get()),
                "use_grating_list": bool(self.use_grating_list.get()),
                "window_geometry": self.window.geometry(),
                "last_spectral_line_file": self.last_spectral_line_file,
                "last_grating_list_file": self.last_grating_list_file
            }

            # безопасная запись (записываем сначала в tmp, затем переименуем)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)

            logger.info("Saved settings to %s", path)
        except Exception as exc:
            logger.warning("Не удалось сохранить настройки %s: %s", path, exc, exc_info=True)

    def _setup_ui(self):
        """Создание элементов интерфейса"""
        # Основной контейнер с использованием grid для корректного распределения областей
        main_container = ttk.Frame(self.window, padding="8")
        main_container.pack(fill='both', expand=True)

        # Настройка колонок: [0] — левая, [1] — таблица, [2] — графики
        main_container.columnconfigure(0, weight=0)
        main_container.columnconfigure(1, weight=1)
        # main_container.columnconfigure(2, weight=0)  # вывод команд — фиксированный размер
        main_container.rowconfigure(0, weight=1)

        # Левая колонка (управление)
        left_frame = ttk.Frame(main_container)
        left_frame.grid(row=0, column=0, sticky='nsw', padx=(0, 10))

        # Центральная колонка (графики)
        center_frame = ttk.Frame(main_container)
        center_frame.grid(row=0, column=1, sticky='nsew')

        # Элементы управления
        self._create_setting_frame(left_frame)
        self._progress_frame(left_frame)
        self._zemax_frame(left_frame)

        # Таблица с оптимальными решетками
        self._create_table(center_frame)

    def _create_table(self, parent):
        """Создает фрейм таблицы с результатами (использует tkintertable)"""
        self.table_frame = ttk.Frame(parent)
        self.table_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Создаем модель таблицы (пока пустую)
        self.table_model = TableModel()
        self.table = TableCanvas(
            self.table_frame,
            model=self.table_model,
            editable=False,  # можно включить редактирование при необходимости
            showtoolbar=True,  # панель с кнопками (сортировка, экспорт и т.п.)
            showstatusbar=True,  # строка состояния
        )
        self.table.fontsize = 8
        self.table.show()
        self.table.autoResizeColumns()

        self.table.bind("<ButtonRelease-1>", self._on_table_select)

    def _disable_cell_editing(self, event):
        # Просто прерываем встроенную обработку редактирования
        return "break"

    def _on_table_select(self, event):
        try:
            row = self.table.get_row_clicked(event)
            if row is None:
                return

            # получаем словарь значений выбранной строки
            record = self.table.model.getRecordAtRow(row)
            # извлекаем оригинальный индекс
            data_index = int(record.get("orig_index", row))

            spectrometer = self.ogfinder.spectrometers[data_index]
            self.active_spectrometer = spectrometer
            if self.active_spectrometer is not None:
                self.zemax_btn.config(state=tk.NORMAL)
                self.draw_echellegramma_btn.config(state=tk.NORMAL)
        except Exception as e:
            logger.warning(f"Ошибка выбора строки: {e}", exc_info=True)

    def _click_button_echellegramma(self):
        try:

            if self.active_spectrometer is not None:
                self._draw_echellegramma(self.active_spectrometer)
        except Exception as e:
            logger.warning(f"Ошибка построения эшеллеграммы: {e}", exc_info=True)

    def _draw_echellegramma(self, spectrometer):
        try:
            # Преобразование к числовым типам и остальная логика остаются без изменений
            lines_in_mm = spectrometer.result.N
            gamma_deg = spectrometer.result.gamma_deg
            grating_cross_tilt_rad = spectrometer.grating.gr_cross_tilt_rad
            k_max = int(spectrometer.result.kmax)
            focal = float(spectrometer.result.f_mm)
            prism_wedge_angle_deg = float(spectrometer.result.prism_deg)
            glass_type = str(self.glass_type.get())
            gap_mm = float(spectrometer.result.gap_mm)
            df_avg = spectrometer.df_avg
            df_prism_min = spectrometer.df_prism_min
            matrix_size = float(self.limit.get()) * 2.0

            # Создание экземпляра расчётчика эшеллеграммы
            ech_drawer = EchellegrammaDrawer(
                spectrometer=spectrometer,
                matrix_size=matrix_size,
            )

            # Вычисление эшеллеграммы
            self.echellegramma_orders = ech_drawer.draw_echellegramma(use_multiprocessing=True)
            if not self.echellegramma_orders:
                logger.debug("⚠️ Не удалось построить эшеллеграмму: нет валидных порядков.")
                return

            # --- Построение графика ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Эшеллеграмма\nN = {lines_in_mm:.1f}, γ = {gamma_deg:.1f}°", fontsize=11)
            ax.set_xlabel("x (мм)")
            ax.set_ylabel("y (мм)")

            # Построение линий порядков
            for order in self.echellegramma_orders:
                ax.plot(
                    [order.x_min_clipped, order.x_max_clipped],
                    [order.y_min_clipped + gap_mm / 2, order.y_max_clipped + gap_mm / 2],
                    color="blue",
                    alpha=0.9,
                    linewidth=1.2,
                )
                x_label = (order.x_min_clipped + order.x_max_clipped) / 2.0
                y_label = (order.y_min_clipped + order.y_max_clipped) / 2.0
                ax.text(
                    x_label,
                    y_label,
                    f"{order.k}",
                    fontsize=8,
                    color="black",
                    ha="center",
                    va="center",
                    alpha=0.8,
                )

            phi2 = grating_groove_tilt_rad(
                prism_wedge_angle_rad=math.radians(prism_wedge_angle_deg),
                prism_tilt_deg=spectrometer.prism.tilt_deg,
                grating_tilt_deg=spectrometer.grating.grating_tilt_deg
            )

            if self.draw_line.get():
                if self.ogfinder.spectra_lines_list is not None:
                    for line in self.ogfinder.spectra_lines_list:
                        x, y = wavelength_to_detector_coords(
                            line[1], k_max, focal, lines_in_mm,
                            math.radians(gamma_deg), grating_cross_tilt_rad,
                            0, math.radians(prism_wedge_angle_deg), df_avg,
                            df_prism_min, glass_type,
                            phi2
                        )
                        y = y + gap_mm / 2
                        ax.scatter(x, y, color="red", s=5)
                        ax.text(x, y + .05, f"{line[0]}, {line[1] if self.draw_line_lambda.get() else ''}", fontsize=6)

            half_limit = float(self.limit.get())
            ax.set_xlim(-half_limit, half_limit)
            ax.set_ylim(0, 2 * half_limit)

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.show()

        except Exception as exc:
            logger.debug(f"❌ Ошибка при построении эшеллеграммы: {exc}")

    def _create_setting_frame(self, parent):
        """Создает фрейм давления"""
        frame = ttk.LabelFrame(parent, text="Параметры оптимизации", padding="5")
        frame.pack(fill='x', pady=5)

        width_entry = 7  # ширина полей для ввода значений

        ttk.Label(frame, text="Полуширина матрицы (мм):").grid(row=0, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.limit, width=width_entry).grid(row=0, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Материал призмы:").grid(row=1, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.glass_type, width=width_entry).grid(row=1, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Ширина щели (мкм):").grid(row=2, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.slit_width, width=width_entry).grid(row=2, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Предел разрешения (нм):").grid(row=3, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.res_limit, width=width_entry).grid(row=3, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Максимальный фокус (мм):").grid(row=4, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.max_focal, width=width_entry).grid(row=4, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Мин. отнош. фокуса к ширине матрицы:").grid(row=5, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.max_ratio, width=width_entry).grid(row=5, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Расстояние между порядками (мм):").grid(row=6, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.min_dist, width=width_entry).grid(row=6, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Макс. потеря линий:").grid(row=width_entry, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.max_lost_line, width=width_entry).grid(row=7, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Минимальная длина волны:").grid(row=8, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.lambda_min, width=width_entry).grid(row=8, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Максимальная длина волны:").grid(row=9, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.lambda_max, width=width_entry).grid(row=9, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Средняя длина волны:").grid(row=10, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.lambda_ctr, width=width_entry).grid(row=10, column=1, padx=5, sticky='w')
        ttk.Label(frame, text="Линий на мм:").grid(row=11, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.lines_in_mm[0], width=width_entry).grid(row=11, column=1, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.lines_in_mm[1], width=width_entry).grid(row=11, column=2, padx=5, sticky='w')
        ttk.Label(frame, text="Угол блеска:").grid(row=12, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.gamma[0], width=width_entry).grid(row=12, column=1, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.gamma[1], width=width_entry).grid(row=12, column=2, padx=5, sticky='w')
        self.use_grating_list_chbtn = ttk.Checkbutton(frame, text='Использовать список решеток',
                                                      variable=self.use_grating_list)
        self.use_grating_list_chbtn.grid(row=13, column=0)
        if self.ogfinder.grating_list is None:
            self.use_grating_list_chbtn.config(state=tk.DISABLED)
        ttk.Label(frame, text="Выводить на эшеллеграмму:").grid(row=14, column=0, padx=5, sticky='w')
        ttk.Checkbutton(frame, text='список линий', variable=self.draw_line).grid(row=14, column=1)
        ttk.Checkbutton(frame, text='длины волн', variable=self.draw_line_lambda).grid(row=14, column=2)

        ttk.Button(frame, text="Загр. список спектр. линий",
                   command=self._load_spectral_line_list).grid(row=15, column=0, padx=5)
        ttk.Button(frame, text="Загр. список дифр. решеток",
                   command=self._load_grating_list).grid(row=15, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.start_btn = ttk.Button(frame, text="Найти оптим. спектрометры", command=self._search_optimal_grating)
        self.start_btn.grid(row=16, column=0, padx=5)
        self.draw_echellegramma_btn = ttk.Button(frame, text="Построить эшеллеграмму",
                                                 command=self._click_button_echellegramma)
        self.draw_echellegramma_btn.grid(row=17, column=0, padx=5)
        if self.active_spectrometer is None:
            self.draw_echellegramma_btn.config(state=tk.DISABLED)

    def _progress_frame(self, parent):
        """Создает фрейм прогрессора"""
        frame = ttk.LabelFrame(parent, text="Статус:", padding="5")
        frame.pack(fill='x', pady=5)

        self.progress = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(padx=20, pady=20)

    def _zemax_frame(self, parent):
        """Создает фрейм модификации файла Земакс"""
        frame = ttk.LabelFrame(parent, text="Изменить файл design.zmx:", padding="1")
        frame.pack(fill='x', pady=1)

        self.zemax_btn = ttk.Button(frame, text="Создать файл *.zmx", command=self._zmx_create)
        self.zemax_btn.grid(row=0, column=0, padx=1)
        if self.active_spectrometer is None:
            self.zemax_btn.config(state=tk.DISABLED)

    def _update_config(self):
        self.config = ConfigOGF(
            lines_in_mm=[self.lines_in_mm[0].get(), self.lines_in_mm[1].get()],
            gamma_deg=[self.gamma[0].get(), self.gamma[1].get()],
            lambda_min=self.lambda_min.get(),
            lambda_max=self.lambda_max.get(),
            lambda_ctr=self.lambda_ctr.get(),
            matrix_size=self.limit.get() * 2,
            glass_type=self.glass_type.get(),
            slit_width=self.slit_width.get(),
            res_limit=self.res_limit.get(),
            max_focal=self.max_focal.get(),
            min_dist_k=self.min_dist.get(),
            max_focal_matrix_ratio=self.max_ratio.get(),
            max_lost_line=self.max_lost_line.get()
        )

    def _search_optimal_grating(self):
        # перед запуском очистим старые окна matplotlib
        plt.close('all')

        self._update_config()
        self.ogfinder.load_config(self.config)

        self.start_btn.config(state=tk.DISABLED)
        self.zemax_btn.config(state=tk.DISABLED)
        self.draw_echellegramma_btn.config(state=tk.DISABLED)

        # # Создаём безопасный callback — он только ставит задачу в mainloop
        # def progress_callback(done: int, total: int):
        #     # schedule update in main thread
        #     try:
        #         # используем self.window (или self.root) — у вас должно быть главное окно
        #         self.window.after(0, lambda: self._update_progressbar(done, total))
        #     except Exception:
        #         # если self.window отсутствует — попытка fallback: ничего не делать
        #         pass

        thread = threading.Thread(target=self._thread_worker, daemon=True)
        thread.start()
        self.check_thread(thread)

        return

    def _thread_worker(self):
        """Рабочий поток — выполняет только расчёт, без GUI."""
        try:
            self.run_search()  # ← здесь не должно быть обращений к Tkinter!
        except Exception as e:  # если ошибка, сообщаем в GUI через after()
            self.window.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        else:
            # если всё ок — вызываем обновление GUI
            self.window.after(0, self._on_search_finished)

    def _update_progressbar(self, done: int, total: int):
        """Обновляет виджет progress в главном потоке."""
        try:
            # инициализация границ (если нужно)
            self.progress["maximum"] = total
            self.progress["value"] = done
            # допустимо обновить интерфейс
            self.progress.update_idletasks()
        except Exception:
            pass

    def _on_search_finished(self):
        """Этот метод выполняется уже в главном потоке."""
        self.start_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Готово", "Поиск оптимальных спектрометров завершён.")

    def _zmx_create(self):
        editor = ZmxEchelleEditor(
            filename="design.zmx",
            spectrometer=self.active_spectrometer,
            config=self.config,
            center_config_idx=1,  # индекс конфигурации в файле, где должна быть центральная (по вашему описанию 7)
            output="design-modified.zmx",
            dry_run=False  # True — только расчёт/логи, не записывать файл
        )

        editor.process()

        messagebox.showinfo("!!! ВНИМАНИЕ !!!", f"Отмасштабируйте схему "
                                                f"на {self.active_spectrometer.focal_mm / 375} \n"
                                                f"Затем выполните оптимизацию.")

    def run_search(self):
        self.ogfinder.search_optimal(use_grating_list=self.use_grating_list.get())


    def check_thread(self, thread):
        """Проверяем поток, чтобы обновлять интерфейс после завершения"""
        if thread.is_alive():
            if hasattr(self.ogfinder, "done") and hasattr(self.ogfinder, "total"):
                done = self.ogfinder.done.value
                total = self.ogfinder.total
                self._update_progressbar(done, total)
            self.window.after(200, lambda: self.check_thread(thread))
        else:
            self.start_btn.config(state=tk.NORMAL)
            if self.ogfinder.spectrometers is not None:
                df = pd.DataFrame([r.result.to_dict(drop_lists=True) for r in self.ogfinder.spectrometers])
                self.optimal_grating_dataframe = df

            self._update_grating_tree()

    def _update_grating_tree(self):
        """Обновляет таблицу tkintertable из DataFrame"""
        if self.optimal_grating_dataframe is None:
            return

        try:
            # создаём копию DataFrame с добавлением исходного индекса
            df = self.optimal_grating_dataframe.copy()
            df.insert(0, "orig_index", range(len(df)))  # добавляем колонку с оригинальными индексами

            # Ограничим число знаков после запятой для красоты
            df = df.round(4)

            # Обновляем модель таблицы напрямую из DataFrame
            self.table_model = TableModel()
            self.table_model.importDict(df.to_dict(orient="index"))

            # удаляем колонку из видимых
            if "orig_index" in self.table_model.columnNames:
                self.table_model.columnNames.remove("orig_index")

            rename_map = {
                "N": "Лин/мм",
                "gamma_deg": "Блеск(°)",
                "kmin": "kmin",
                "kmax": "kmax",
                "f_mm": "Фокус(мм)",
                "prism_deg": "Угол пр.",
                "ratio_f_det": "Отн. f/Dm",
                "spectral_full_nm": "Весь(нм)",
                "spectral_visible_nm": "Видим.(нм)",
                "spectral_lost_nm": "Потеря(нм)",
                "num_lost_lines": "Потеря(лин)",
                "loss_pct": "Потеря(%)",
                "gap_mm": "Зазор(мм)",
                # ... добавь остальные
            }

            # Присваиваем новой модели существующую таблицу
            self.table.updateModel(self.table_model)
            for i, colName in enumerate(self.table.model.columnNames):
                if colName in rename_map:
                    self.table.model.relabel_Column(i, rename_map[colName])
            self.table.redrawTable()
            self._auto_resize_columns()


        except Exception as e:
            logger.exception(f"Ошибка при обновлении таблицы tkintertable: {e}")

    def _auto_resize_columns(self):
        """Автоматически подбирает ширину колонок под содержимое и заголовки"""
        from tkinter import font

        try:
            model = self.table.model
            data = model.data

            # Получаем список колонок
            if hasattr(model, "columnlabels"):
                columns = model.columnlabels
            elif hasattr(model, "columnNames"):
                columns = model.columnNames
            elif hasattr(model, "columns"):
                columns = list(model.columns.keys())
            elif len(data) > 0:
                # берем по первой строке, если структура есть
                columns = list(next(iter(data.values())).keys())
            else:
                return

            # Получаем ширину символа текущего шрифта
            f = font.nametofont("TkDefaultFont")
            char_width = f.measure("%")

            column_widths = {}

            for col in columns:
                # вычисляем макс. длину в колонке + заголовок
                lengths = [len(str(columns[col]))]
                for row in data.values():
                    val = row.get(col, "")
                    lengths.append(len(str(val)))
                max_len = max(lengths)

                # минимальная ширина — 2 символа
                width_px = max(2 * char_width, max_len * char_width)
                column_widths[col] = width_px

            # применяем новые ширины
            model.columnwidths.update(column_widths)
            self.table.redrawTable()

        except Exception as e:
            logger.warning(f"⚠️ Ошибка авторасчета ширины: {e}")

    def _load_spectral_line_list(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel файлы", "*.xlsx"), ("Все файлы", "*.*")],
            title="Загрузить файл списка спектральных линий"
        )

        if file_path:
            try:
                self.ogfinder.load_spectra_lines_list_from_excel(file_path)
                self.last_spectral_line_file = file_path
            except Exception as e:
                logger.warning("Ошибка чтения файла со списком спектральных линий: %s", e, exc_info=True)


    def _load_grating_list(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel файлы", "*.xlsx"), ("Все файлы", "*.*")],
            title="Загрузить файл списка дифракционных решеток"
        )

        if file_path:
            try:
                self.ogfinder.load_grating_list_from_excel(file_path)
                self.last_grating_list_file = file_path
            except Exception as e:
                logger.warning("Ошибка чтения файла со списком спектральных линий: %s", e, exc_info=True)

    def on_close(self):
        """Обработчик закрытия окна — сохраняем настройки и закрываем окно."""
        try:
            self._save_settings()
        except Exception:
            # не мешаем закрытию даже если сохранение упало
            logger.exception("Ошибка при сохранении настроек при закрытии")
        finally:
            # уничтожаем окно
            try:
                self.window.destroy()
            except Exception:
                pass

    def run(self):
        """Запускает главный цикл приложения"""
        self.window.mainloop()
