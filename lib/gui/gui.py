"""Модуль графического интерфейса"""
import math

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import threading
import tkinter as tk
from tkinter import *
from tkinter import Tk, BooleanVar, StringVar, IntVar, Menu, DoubleVar
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lib.echele.optimalGratingFinder import OptimalGratingFinder
from lib.echele.echelegrammaDrawer import EchelegrammaDrawer
from lib.echele.echeleMath import (
    find_orders_range, wavelength_to_detector_coords,
    grating_groove_tilt_rad
)


class EcheleGUI:
    """Класс графического интерфейса для управления устройством"""

    def __init__(self, ogfinder: OptimalGratingFinder):
        self.ogfinder = ogfinder
        self.window = Tk()
        self._setup_window()
        self._init_variables()
        self._setup_ui()

    def _setup_window(self):
        """Настройка основного окна"""
        self.window.title("Эшелле решетки")
        self.window.geometry("1550x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_variables(self):
        """Инициализация переменных интерфейса"""

        self.lines_in_mm = [IntVar(value=80), IntVar(value=120)]
        self.gamma = [IntVar(value=40), IntVar(value=80)]
        self.limit = DoubleVar(value=10.4)  # полуширина матрицы (мм)
        self.glass_type = StringVar(value="CaF")  # 1 = BaF, 2 = CaF
        self.slit_width = IntVar(value=21)  # мкм
        self.res_limit = DoubleVar(value=6.7e-3)  # 4 пм = 0.004 нм
        self.max_focal = IntVar(value=800)  # мм
        self.max_ratio = IntVar(value=10)  # f / (2*LIMIT) < 15
        self.min_dist = DoubleVar(value=0.08)  # расстояние между порядками
        self.max_lost_line = IntVar(value=0)  # максимальная возможная потеря из списка спектральных линий. 0 - без ограничений

        self.lambda_min = IntVar(value=167)  # нм
        self.lambda_max = IntVar(value=780)  # нм
        self.lambda_ctr = IntVar(value=200)  # нм

        self.draw_line = BooleanVar(value=True)     # выводить список спектральных линий на эшелеграмме
        self.draw_line_lambda = BooleanVar(value=True)     # подписывать длины волн

        self.optimal_grating_dataframe = None

        self.progress = None

    def _setup_ui(self):
        """Создание элементов интерфейса"""
        # Основной контейнер с использованием grid для корректного распределения областей
        main_container = ttk.Frame(self.window, padding="8")
        main_container.pack(fill='both', expand=True)

        # Настройка колонок: [0] — левая, [1] — таблица, [2] — графики
        main_container.columnconfigure(0, weight=0)
        main_container.columnconfigure(1, weight=1)
        #main_container.columnconfigure(2, weight=0)  # вывод команд — фиксированный размер
        main_container.rowconfigure(0, weight=1)

        # Левая колонка (управление)
        left_frame = ttk.Frame(main_container)
        left_frame.grid(row=0, column=0, sticky='nsw', padx=(0, 10))

        # Центральная колонка (графики)
        center_frame = ttk.Frame(main_container)
        center_frame.grid(row=0, column=1, sticky='nsew')

        # Правая колонка (вывод команд)
        # right_frame = ttk.Frame(main_container)
        # right_frame.grid(row=0, column=2, sticky='nse', padx=(10, 0))

        # Элементы управления
        self._create_setting_frame(left_frame)
        self._progress_frame(left_frame)

        # Таблица с оптимальными решетками
        self.create_table(center_frame)

    def _create_graphs_frame(self, parent):
        """Создает фрейм с графиками"""
        frame = ttk.LabelFrame(parent, text="Эшеллеграмма", padding="5")
        frame.pack(fill='both', expand=True, pady=5)

        # Убедимся, что фигура уже создана
        if not hasattr(self, 'canvas'):
            self._init_graphs(frame)

        # Размещаем canvas так, чтобы он занимал всё пространство фрейма
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Подключаем событие отрисовки
        self.canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event):
        """Обработчик события отрисовки для сохранения фона"""
        # self.ax1_background = self.canvas.copy_from_bbox(self.ax1.bbox)
        # self.ax2_background = self.canvas.copy_from_bbox(self.ax2.bbox)

    def create_table(self, parent):
        """Создает фрейм таблицы с результатами"""
        # Таблица добавленных элементов со свойствами
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Вертикальный скроллбар
        y_scroll = ttk.Scrollbar(tree_frame, orient=VERTICAL)
        y_scroll.pack(side=RIGHT, fill=Y)
        x_scroll = ttk.Scrollbar(tree_frame, orient=HORIZONTAL)
        x_scroll.pack(side=BOTTOM, fill=X)

        columns = ("N", "gamma_deg", "kmin", "kmax", "f_mm", "prism_deg", "gap_mm",
                   "ratio_f_det", "spectral_full_nm", "spectral_visible_nm", "spectral_lost_nm",
                   "num_lost_lines", "loss_pct")

        self.grating_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            yscrollcommand=y_scroll.set,
            xscrollcommand = x_scroll.set
        )
        self.grating_tree.pack(fill=BOTH, expand=True)

        # Настройка скроллбаров
        y_scroll.config(command=self.grating_tree.yview)
        x_scroll.config(command=self.grating_tree.xview)

        for column in columns:
            self.grating_tree.heading(column, text=column)
            self.grating_tree.column(column, width=80, anchor=CENTER)

        self.grating_tree.bind("<<TreeviewSelect>>", self._on_element_select)

    def _on_element_select(self, event):
        selected = self.grating_tree.selection()
        selected_id = self.grating_tree.index(selected[0])
        if not selected:
            return

        try:
            # Извлечение данных о решётке
            grating_values = self.grating_tree.item(selected[0], "values")
            (
                lines_in_mm_str,
                gamma_deg_str,
                k_min_str,
                k_max_str,
                focal_str,
                prism_wedge_angle_deg_str,
                gap_mm,
                *_,
            ) = grating_values

            spectrometr = self.ogfinder.spectrometers[selected_id]

            # Преобразование к числовым типам
            lines_in_mm = spectrometr.result.N
            gamma_deg = spectrometr.result.gamma_deg
            k_min = int(spectrometr.result.kmin)
            k_max = int(spectrometr.result.kmax)
            focal = float(spectrometr.result.f_mm)
            prism_wedge_angle_deg = float(spectrometr.result.prism_deg)
            glass_type = str(self.glass_type.get())
            gap_mm = float(spectrometr.result.gap_mm)
            df_avg = spectrometr.df_avg
            df_prism_min = spectrometr.df_prism_min

            dx = 0
            dy = gap_mm
            matrix_size = float(self.limit.get()) * 2.0

            # Создание экземпляра расчётчика эшеллеграммы
            ech_drawer = EchelegrammaDrawer(
                spectrometr=spectrometr,
                matrix_size=matrix_size,
            )

            # Вычисление эшеллеграммы
            orders = ech_drawer.draw_echelegramma(use_multiprocessing=True)
            if not orders:
                print("⚠️ Не удалось построить эшеллеграмму: нет валидных порядков.")
                return

            # --- Построение графика ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Эшеллеграмма\nN = {lines_in_mm:.1f}, γ = {gamma_deg:.1f}°", fontsize=11)
            ax.set_xlabel("x (мм)")
            ax.set_ylabel("y (мм)")

            # Построение линий порядков
            for order in orders:
                ax.plot(
                    [order.x_min_clipped, order.x_max_clipped],
                    [order.y_min_clipped, order.y_max_clipped],
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
                prism_tilt_deg=spectrometr.prism.tilt_deg,
                grating_tilt_deg=spectrometr.grating.grating_tilt_deg
            )

            if self.draw_line.get():
                for line in self.ogfinder.spectra_lines_list:
                    x, y = wavelength_to_detector_coords(line[1], k_max, focal, lines_in_mm, math.radians(gamma_deg),
                                                         0, np.radians(prism_wedge_angle_deg), df_avg,
                                                         df_prism_min, glass_type,
                                                         phi2)
                    ax.scatter(x, y, color="red", s=5)
                    ax.text(x, y + .05, f"{line[0]}, {line[1] if self.draw_line_lambda.get() else ''}", fontsize=6)

            # Ограничения области отображения
            half_limit = float(self.limit.get())
            ax.set_xlim(-half_limit, half_limit)
            ax.set_ylim(0, 2 * half_limit)

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.show()

        except Exception as exc:
            import traceback
            print(f"❌ Ошибка при построении эшеллеграммы: {exc}")
            traceback.print_exc()

    def _update_graphs(self):
        """Оптимизированное обновление графиков"""
        if not (self.receive_new_temperature_data or self.receive_new_pressure_data):
            return

        try:
            redraw_full = False
            # Обновляем данные линий
            if self.receive_new_temperature_data and len(self.temp_data['value']) > 0:
                self.temp_line.set_data(range(len(self.temp_data['value'])), self.temp_data['value'])
                old_ylim = self.ax1.get_ylim()
                self.ax1.relim()
                self.ax1.autoscale_view(scalex=False, scaley=True)
                new_ylim = self.ax1.get_ylim()
                # Проверяем: изменились ли границы Y
                if old_ylim != new_ylim:
                    redraw_full = True
                else:
                    redraw_full = False

            if self.receive_new_pressure_data and len(self.pressure_data['value']) > 0:
                self.pressure_line.set_data(range(len(self.pressure_data['value'])), self.pressure_data['value'])
                old_ylim = self.ax2.get_ylim()
                self.ax2.relim()
                self.ax2.autoscale_view(scalex=False, scaley=True)
                new_ylim = self.ax2.get_ylim()
                # Проверяем: изменились ли границы Y
                if old_ylim != new_ylim:
                    redraw_full = True
                else:
                    redraw_full = False

            if self.receive_new_position_data and len(self.position_data['value']) > 0:
                self.position_line.set_data(range(len(self.position_data['value'])), self.position_data['value'])
                old_ylim = self.ax3.get_ylim()
                self.ax3.relim()
                self.ax3.autoscale_view(scalex=False, scaley=True)
                new_ylim = self.ax3.get_ylim()
                # Проверяем: изменились ли границы Y
                if old_ylim != new_ylim:
                    redraw_full = True
                else:
                    redraw_full = False

            # Первая отрисовка - сохраняем фон
            if self.ax1_background is None or redraw_full:
                self.temp_line.set_animated(True)
                self.pressure_line.set_animated(True)
                self.position_line.set_animated(True)
                self.fig.canvas.draw()
                self.ax1_background = self.canvas.copy_from_bbox(self.ax1.bbox)
                self.ax2_background = self.canvas.copy_from_bbox(self.ax2.bbox)
                self.ax3_background = self.canvas.copy_from_bbox(self.ax3.bbox)


            # Последующие обновления с blitting
            self.canvas.restore_region(self.ax1_background)
            self.ax1.draw_artist(self.temp_line)
            self.canvas.restore_region(self.ax2_background)
            self.ax2.draw_artist(self.pressure_line)
            self.canvas.restore_region(self.ax3_background)
            self.ax3.draw_artist(self.position_line)

            # Обновляем только измененные области
            self.canvas.blit(self.ax1.bbox)
            self.canvas.blit(self.ax2.bbox)
            self.canvas.blit(self.ax3.bbox)

        except Exception as e:
            self.append_command_log(f"Ошибка обновления графиков: {e}")
            # При ошибке перерисовываем полностью
            self.canvas.draw()
        finally:
            self.receive_new_temperature_data = False
            self.receive_new_pressure_data = False
            self.receive_new_position_data = False

    def _create_setting_frame(self, parent):
        """Создает фрейм давления"""
        frame = ttk.LabelFrame(parent, text="Параметры оптимизации", padding="5")
        frame.pack(fill='x', pady=5)

        width_entry = 7     # ширина полей для ввода значений

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
        ttk.Label(frame, text="Макс. отнош. фокуса к ширине матрицы:").grid(row=5, column=0, padx=5, sticky='w')
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
        ttk.Checkbutton(frame, text='Выводить список линий', variable=self.draw_line).grid(row=13, column=0)
        ttk.Checkbutton(frame, text='Выводить длины волн', variable=self.draw_line_lambda).grid(row=13, column=1)

        self.start_btn = ttk.Button(frame, text="Найти оптим. решетки", command=self._search_optimal_grating)
        self.start_btn.grid(row=14, column=0, padx=5)

    def _progress_frame(self, parent):
        """Создает фрейм прогрессора"""
        frame = ttk.LabelFrame(parent, text="Статус:", padding="5")
        frame.pack(fill='x', pady=5)

        self.progress = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(padx=20, pady=20)

    def _search_optimal_grating(self):
        self.ogfinder.lines_in_mm = [self.lines_in_mm[0].get(), self.lines_in_mm[1].get()]
        self.ogfinder.gamma_rad = np.radians([self.gamma[0].get(), self.gamma[1].get()])
        self.ogfinder.lambda_min = self.lambda_min.get()
        self.ogfinder.lambda_max = self.lambda_max.get()
        self.ogfinder.lamda_ctr = self.lambda_ctr.get()
        self.ogfinder.matrix_size = self.limit.get() * 2
        self.ogfinder.glass_type = self.glass_type.get()
        self.ogfinder.slit_width = self.slit_width.get()
        self.ogfinder.res_limit = self.res_limit.get()
        self.ogfinder.max_focal = self.max_focal.get()
        self.ogfinder.min_dist_k = self.min_dist.get()
        self.ogfinder.max_focal_matrix_ratio = self.max_ratio.get()
        self.ogfinder.max_lost_line = self.max_lost_line.get()

        self.start_btn: ttk.Button
        self.start_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_search, daemon=False)
        thread.start()
        self.check_thread(thread)

        return

    def run_search(self):
        # здесь вызывается твоя функция
        self.ogfinder.search_optimal(progress=self.progress)

    def check_thread(self, thread):
        """Проверяем поток, чтобы обновлять интерфейс после завершения"""
        if thread.is_alive():
            self.window.after(200, lambda: self.check_thread(thread))
        else:
            self.start_btn.config(state=tk.NORMAL)
            if self.ogfinder.spectrometers is not None:
                df = pd.DataFrame([r.result.to_dict(drop_lists=True) for r in self.ogfinder.spectrometers])
                self.optimal_grating_dataframe = df

            self.update_grating_tree()

    def update_grating_tree(self):
        """Обновление таблицы Treeview с текущими добавленными элементами"""
        if (not hasattr(self, 'grating_tree') or self.grating_tree is None or
                self.optimal_grating_dataframe is None):
            return  # Выходим, если элемент еще не создан

        for item in self.grating_tree.get_children():
            self.grating_tree.delete(item)

        for val in self.optimal_grating_dataframe.values:
            self.grating_tree.insert("", END, values=val.tolist())

    def _add_context_menu(self, widget):
        """Добавляет контекстное меню с возможностью копирования"""
        menu = Menu(widget, tearoff=0)
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))

        def show_menu(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-3>", show_menu)  # ПКМ для Windows и Linux

    def on_close(self):
        """Обработчик закрытия окна"""
        # self.logger.flush()  # Сохраняем данные перед выходом
        # self.controller.disconnect()
        self.window.destroy()

    def run(self):
        """Запускает главный цикл приложения"""
        self.window.mainloop()