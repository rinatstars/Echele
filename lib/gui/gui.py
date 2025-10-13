"""Модуль графического интерфейса"""

import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tkinter import *
from tkinter import Tk, BooleanVar, StringVar, IntVar, Menu, DoubleVar
from tkinter import ttk
from tkinter import scrolledtext

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lib.echele.optimalGratingFinder import OptimalGratingFinder
from lib.echele.echelegrammaDrawer import EchelegrammaDrawer
from collections import deque
#from logger import DataLogger  # Добавляем импорт


matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('TkAgg')  # Явно указываем бэкенд

class EcheleGUI:
    """Класс графического интерфейса для управления устройством"""

    def __init__(self, ogfinder: OptimalGratingFinder):
        self.ogfinder = ogfinder
        self.window = Tk()
        self._setup_window()
        self._init_variables()
        self._setup_ui()
        #self._start_background_tasks()
        #self.logger = DataLogger(log_interval=60)  # Создаем экземпляр логгера
        #self.controller.init_func_time_culc(self._update_interval_upd_data)

    def _setup_window(self):
        """Настройка основного окна"""
        self.window.title("Эшелле решетки")
        self.window.geometry("1550x800")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_variables(self):
        """Инициализация переменных интерфейса"""
        # self.status_vars = {
        #     'STAB': BooleanVar(),
        #     'OPEN': BooleanVar(),
        #     'CLOSE': BooleanVar(),
        #     'POSITION': BooleanVar(),
        #     'KEY STAB': BooleanVar(),
        #     'KEY OPEN': BooleanVar(),
        #     'KEY CLOSE': BooleanVar(),
        #     'ERROR': BooleanVar(),
        #     'RESET': BooleanVar(),
        #     'PING': BooleanVar(),
        # }
        # self.measured_pressure_var = StringVar(value="--- Pa")
        # self.set_pressure_var = StringVar(value="0")
        # self.temperature_var = StringVar(value="--- °C")
        # self.position_var = IntVar(value=0)
        # self.position_var_set = IntVar(value=100)
        # self.position_text_var = StringVar(value="Позиция изм.: 0")
        # self.position_text_var_set = StringVar(value="Позиция уст.: 100")
        # self.receive_new_temperature_data = False
        # self.receive_new_pressure_data = False
        # self.receive_new_position_data = False
        # self.receive_new_status_data = False
        # self.last_log_time = time.time()
        # self.calc_speed = False
        # self.text_press = "Давление"
        # self.log_enable = BooleanVar(value=False)
        # self.interval_polling = StringVar(value="Обновление окна: ---мс")
        # self.interval_upd_data = StringVar(value="Обновление данных: ---мс")

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

        self.optimal_grating_dataframe = None

    def _init_graphs(self, frame):
        # Данные для графиков
        # self.max_points = 100  # Фиксированное количество точек
        # self.temp_data = {'time': deque(maxlen=self.max_points), 'value': deque(maxlen=self.max_points)}
        # self.pressure_data = {'time': deque(maxlen=self.max_points), 'value': deque(maxlen=self.max_points)}
        # self.position_data = {'time': deque(maxlen=self.max_points), 'value': deque(maxlen=self.max_points)}

        # Настройка фигуры
        self.fig = Figure(figsize=(6, 5), dpi=80)
        self.fig.set_tight_layout(True)

        # Настройка осей
        self.ax1 = self.fig.add_subplot(111)
        # self.ax2 = self.fig.add_subplot(312)
        # self.ax3 = self.fig.add_subplot(313)

        # Инициализация линий графиков
        self.temp_line, = self.ax1.plot([], [], 'b-')
        # self.pressure_line, = self.ax2.plot([], [], 'b-', label='Давление')
        # self.position_line, = self.ax3.plot([], [], 'g-', label='Позиция')

        # Настройка осей
        self.ax1.set_title('Эшеллеграмма')
        #self.ax1.set_xlim(0, self.max_points - 1)
        self.ax1.grid(False)
        self.ax1.legend(loc='upper right')
        # self.ax2.set_title('Давление (Pa)')
        # self.ax2.set_xlim(0, self.max_points - 1)
        # self.ax2.grid(True)
        # self.ax2.legend(loc='upper right')
        # self.ax3.set_title('Позиция')
        # self.ax3.set_xlim(0, self.max_points - 1)
        # self.ax3.grid(True)
        # self.ax3.legend(loc='upper right')

        # Инициализация canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()

        # Фон для blitting нужно сохранять после первого отображения
        self.ax1_background = None
        self.ax2_background = None
        self.ax3_background = None

    def _setup_ui(self):
        """Создание элементов интерфейса"""
        # Основной контейнер с использованием grid для корректного распределения областей
        main_container = ttk.Frame(self.window, padding="30")
        main_container.pack(fill='both', expand=True)

        # Настройка колонок: [0] — левая, [1] — таблица, [2] — графики
        main_container.columnconfigure(0, weight=0)
        main_container.columnconfigure(1, weight=1)
        main_container.columnconfigure(2, weight=0)  # вывод команд — фиксированный размер
        main_container.rowconfigure(0, weight=1)

        # Левая колонка (управление)
        left_frame = ttk.Frame(main_container)
        left_frame.grid(row=0, column=0, sticky='nsw', padx=(0, 10))

        # Центральная колонка (графики)
        center_frame = ttk.Frame(main_container)
        center_frame.grid(row=0, column=1, sticky='nsew')

        # Правая колонка (вывод команд)
        right_frame = ttk.Frame(main_container)
        right_frame.grid(row=0, column=2, sticky='nse', padx=(10, 0))

        # Элементы управления
        self._create_setting_frame(left_frame)
        # self._create_temperature_frame(left_frame)
        # self._create_position_frame(left_frame)
        # self._create_pressure_frame(left_frame)
        # self._create_command_frame(left_frame)
        # self._create_log_frame(left_frame)

        # Таблица с оптимальными решетками
        self.create_table(center_frame)

        # Графики
        #self._create_graphs_frame(right_frame)
        #self._create_ping_frame(right_frame)

        # Элемент вывода команд (правая колонка)
        #self.command_output = scrolledtext.ScrolledText(center_frame, width=40, height=30,
        #                                                state='normal', wrap='word')
        #self.command_output.pack(fill='both', expand=True)
        # Запрет ввода вручную
        #self.command_output.bind("<Key>", lambda e: "break")
        # Добавляем контекстное меню
        #self._add_context_menu(self.command_output)

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
            ax.set_title(f"Эшеллеграмма\nN = {lines_in_mm:.0f}, γ = {gamma_deg:.1f}°", fontsize=11)
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

            # vis_line_list, _ = self.ogfinder.vis_los_lines(self.grating_tree.index(selected[0]))
            # for line in vis_line_list:


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


    # def _update_status(self):
    #     """Обновляет статусные флаги"""
    #     while not self.controller.status_queue.empty():
    #         address, value = self.controller.status_queue.get()
    #         if address == REG_STATUS:
    #             for i, (name, var) in enumerate(self.status_vars.items()):
    #                 var.set(bool(value & (1 << i)))

    # def _update_position(self):
    #     """Обновляет позицию заслонки (32-битное значение)"""
    #     position_lo = None
    #     position_hi = None
    #
    #     while not self.controller.position_queue_LO.empty() and not self.controller.position_queue_HI.empty():
    #         address, value = self.controller.position_queue_LO.get()
    #         position_lo = value
    #         address, value = self.controller.position_queue_HI.get()
    #         position_hi = value
    #
    #     if position_lo is not None and position_hi is not None:
    #         position = (position_hi << 16) | position_lo
    #         self.position_var.set(position)
    #         self.position_data['value'].append(position)
    #         self.position_text_var.set(f"Позиция изм.: {position}")
    #
    #         self.receive_new_position_data = True
    #         #print(f'pos_lo: {position_lo}, pos_hi: {position_hi}, pos: {position}')

    # def _update_temperature(self):
    #     """Обновляет показания температуры"""
    #     max_updates = 10  # Ограничиваем количество обновлений за один вызов
    #     updates = 0
    #
    #     while not self.controller.temperature_queue.empty() and updates < max_updates:
    #         try:
    #             address, value = self.controller.temperature_queue.get_nowait()
    #             if address == REG_TEMPERATURE:
    #                 temp_c = value / 10.0
    #                 self.temp_data['value'].append(temp_c)
    #                 self.temperature_var.set(f"{temp_c:.1f} °C")
    #                 self.receive_new_temperature_data = True
    #                 updates += 1
    #         except:
    #             break

    # def _update_pressure(self):
    #     """Обновляет показания давления"""
    #     max_updates = 10
    #     updates = 0
    #
    #     while not self.controller.measured_pressure_queue.empty() and updates < max_updates:
    #         try:
    #             address, value = self.controller.measured_pressure_queue.get_nowait()
    #             if address == REG_MEASURED_PRESSURE:
    #                 pressure = value / 10.0
    #                 self.pressure_data['value'].append(pressure)
    #                 self.measured_pressure_var.set(f"{pressure:.1f} Pa")
    #                 self.receive_new_pressure_data = True
    #                 updates += 1
    #         except:
    #             break


    # def _update_data(self):
    #     """Обновляет все данные из очередей"""
    #     try:
    #         self._update_status()
    #         self._update_temperature()
    #         self._update_position()
    #         self._update_pressure()
    #     except Exception as e:
    #         self.append_command_log(f"Ошибка обновления интерфейса: {e}")

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
    #
    # def _update_interval_upd_data(self, interval):
    #     self.interval_upd_data.set(f"Обновление данных: {interval}мс")
    #
    # def _create_ping_frame(self, parent):
    #     frame = ttk.LabelFrame(parent, text="Связь", padding="5")
    #     frame.pack(fill='x', pady=5)
    #
    #     ttk.Label(frame, textvariable=self.interval_polling).grid(row=0, column=0, padx=5, sticky='w')
    #     ttk.Label(frame, textvariable=self.interval_upd_data).grid(row=0, column=1, padx=5, sticky='w')
    #
    # def _create_status_frame(self, parent):
    #     """Создает фрейм статуса"""
    #     frame = ttk.LabelFrame(parent, text="Статус", padding="5")
    #     frame.pack(fill='x', pady=5)
    #
    #     n = 0
    #     for name, var in self.status_vars.items():
    #         if n < 5:
    #             cb = ttk.Checkbutton(frame, text=name, variable=var, state='disabled')
    #             cb.grid(row=n, column=0, padx=5, sticky='w')
    #             n += 1
    #         else:
    #             cb = ttk.Checkbutton(frame, text=name, variable=var, state='disabled')
    #             cb.grid(row=n-5, column=1, padx=5, sticky='w')
    #             n += 1
    #
    #
    # def _create_temperature_frame(self, parent):
    #     """Создает фрейм температуры"""
    #     frame = ttk.LabelFrame(parent, text="Температура", padding="5")
    #     frame.pack(fill='x', pady=5)
    #     ttk.Label(frame, textvariable=self.temperature_var, font=('Arial', 12)).pack()
    #
    # def _create_position_frame(self, parent):
    #     """Создает фрейм позиции"""
    #     frame = ttk.LabelFrame(parent, text="Позиция заслонки", padding="5")
    #     frame.pack(fill='x', pady=5)
    #     ttk.Label(frame, textvariable=self.position_text_var).grid(row=0, column=0, padx=5, sticky='w')
    #     ttk.Label(frame, textvariable=self.position_text_var_set).grid(row=0, column=1, padx=5, sticky='w')
    #     ttk.Scale(frame, variable=self.position_var_set, from_=99, to=1000, length=300, command=self._set_position_var).grid( # 4294967295
    #         row=1, column=0, columnspan=2, padx=5, sticky='w'
    #     )
    #     ttk.Button(frame, text="Применить", command=self._set_position).grid(
    #         row=2, column=0, columnspan=2, padx=5, sticky='n'
    #     )
    #
    # def _change_speed_press(self):
    #     if self.calc_speed:
    #         self.text_press = "Скорость"
    #     else:
    #         self.text_press = "Давление"

    def _create_setting_frame(self, parent):
        """Создает фрейм давления"""
        frame = ttk.LabelFrame(parent, text="Параметры оптимизации", padding="5")
        frame.pack(fill='x', pady=5)

        width_entry = 7

        ttk.Label(frame, text="Полуширина матрицы (мм):").grid(row=0, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.limit, width=width_entry).grid(row=0, column=1, padx=5)
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
        ttk.Label(frame, text="Угол призмы:").grid(row=12, column=0, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.gamma[0], width=width_entry).grid(row=12, column=1, padx=5, sticky='w')
        ttk.Entry(frame, textvariable=self.gamma[1], width=width_entry).grid(row=12, column=2, padx=5, sticky='w')

        # ttk.Label(frame, text="Уставка:").grid(row=1, column=0, padx=5, sticky='w')
        # self.pressure_spinbox = ttk.Spinbox(
        #     frame, from_=0, to=10000, textvariable=self.set_pressure_var, width=10
        # )
        # self.pressure_spinbox.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Найти оптим. решетки", command=self._search_optimal_grating).grid(row=13, column=0, padx=5)
        # ttk.Button(frame, text="Применить", command=self._set_pressure).grid(row=1, column=3, padx=5)

        # cb = ttk.Checkbutton(frame, text="Скорость", variable=self.calc_speed, command=self._change_speed_press)
        # cb.grid(row=2, column=0, padx=5, sticky='w')

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

        self.ogfinder.search_optimal()

        if self.ogfinder.spectrometers is not None:
            df = pd.DataFrame([r.result.to_dict(drop_lists=True) for r in self.ogfinder.spectrometers])
            self.optimal_grating_dataframe = df

        self.update_grating_tree()

        return

    def update_grating_tree(self):
        """Обновление таблицы Treeview с текущими добавленными элементами"""
        if (not hasattr(self, 'grating_tree') or self.grating_tree is None or
                self.optimal_grating_dataframe is None):
            return  # Выходим, если элемент еще не создан

        for item in self.grating_tree.get_children():
            self.grating_tree.delete(item)

        for val in self.optimal_grating_dataframe.values:
            self.grating_tree.insert("", END, values=val.tolist())

        print()

    # def _create_command_frame(self, parent):
    #     """Создает фрейм команд"""
    #     frame = ttk.LabelFrame(parent, text="Команды", padding="5")
    #     frame.pack(fill='x', pady=10)
    #
    #     ttk.Button(frame, text="СТАРТ", command=lambda: self._send_command(REG_COMMAND, CMD_START)).grid(
    #         row=0, column=0, padx=5, pady=2)
    #     ttk.Button(frame, text="СТОП", command=lambda: self._send_command(REG_COMMAND, CMD_STOP)).grid(
    #         row=0, column=1, padx=5, pady=2)
    #     ttk.Button(frame, text="СОХР.FLASH", command=lambda: self._send_command(REG_COMMAND, CMD_SAVE_FLASH)).grid(
    #         row=0, column=2, padx=5, pady=2)
    #
    #     ttk.Button(frame, text="ОТКРЫТО", command=lambda: self._send_command(REG_COMMAND, CMD_OPEN)).grid(
    #         row=1, column=0, padx=5, pady=2)
    #     ttk.Button(frame, text="ЗАКРЫТО", command=lambda: self._send_command(REG_COMMAND, CMD_CLOSE)).grid(
    #         row=1, column=1, padx=5, pady=2)
    #     ttk.Button(frame, text="СРЕДНЕЕ", command=self._set_middle_position).grid(
    #         row=1, column=2, padx=5, pady=2)
    #
    #     ttk.Button(frame, text="ПОЗИЦИЯ", command=lambda: self._send_command(REG_COMMAND, CMD_POSITION)).grid(
    #         row=2, column=0, padx=5, pady=2)
    #
    #     ttk.Button(frame, text="ЗВУК", command=lambda: self._send_command(REG_COMMAND, CMD_SOUND)).grid(
    #         row=2, column=1, padx=5, pady=2)
    #
    #     for i in range(3):
    #         frame.grid_columnconfigure(i, weight=1)
    #
    # def _create_log_frame(self, parent):
    #     """Создает фрейм управления логом"""
    #     frame = ttk.LabelFrame(parent, text="Лог", padding="5")
    #     frame.pack(fill='x', pady=10)
    #
    #     ttk.Button(frame, text="СТАРТ", command=lambda: self._start_log(True)).grid(
    #         row=0, column=0, padx=5, pady=2)
    #     ttk.Button(frame, text="СТОП", command=lambda: self._start_log(False)).grid(
    #         row=0, column=1, padx=5, pady=2)
    #     ttk.Button(frame, text="20 изм", command=lambda: self._rec_to_log(20)).grid(
    #         row=0, column=2, padx=5, pady=2)
    #
    #     cb = ttk.Checkbutton(frame, text="Запущен", variable=self.log_enable, state='disabled')
    #     cb.grid(row=1, column=0, padx=5, sticky='w')
    #
    #     for i in range(3):
    #         frame.grid_columnconfigure(i, weight=1)
    #
    # def _start_background_tasks(self):
    #     """Оптимизированный планировщик задач"""
    #     start_time = time.time()
    #
    #     #self.controller.start_polling(one_poll=True)
    #
    #     # Обновляем данные
    #     self._update_data()
    #
    #     # Обновляем графики только если есть новые данные
    #     if self.receive_new_temperature_data or self.receive_new_pressure_data:
    #         self._update_graphs()
    #
    #     # Логируем данные (реже)
    #     if time.time() - self.last_log_time >= 0.5 and self.log_enable.get():  # Раз в секунду
    #         self._log_data()
    #         self.last_log_time = time.time()
    #
    #     # Динамически регулируем интервал
    #     processing_time = time.time() - start_time
    #     next_interval = max(2, int(processing_time * 1000 * 1.1))  # +10% к времени обработки
    #     self.interval_polling.set(f"Обновление окна: {int(next_interval)}мс")
    #
    #     if self.window.winfo_exists():
    #         self.window.after(next_interval, self._start_background_tasks)
    #
    # def _check_connection(self):
    #     """Проверяет соединение с устройством"""
    #     if not self.controller._ensure_connection():
    #         self.append_command_log("Предупреждение: проблемы с соединением")
    #
    # def _send_command(self, register, value):
    #     """Отправляет команду устройству"""
    #     if self.controller.write_register(register, value):
    #         self.append_command_log(f"Команда отправлена: регистр 0x{register:02X}, значение 0x{value:04X}")
    #
    # def _set_pressure(self):
    #     """Устанавливает давление"""
    #     try:
    #         value = int(float(self.set_pressure_var.get()) * 10)
    #         # status = self.controller.read_register(REG_STATUS)
    #         # if status is None:
    #         #     return
    #
    #         # was_stab = False #status & 0x01
    #         # if was_stab:
    #         #     self.controller.write_register(REG_COMMAND, CMD_STOP)
    #
    #         if self.controller.write_register(REG_SET_PRESSURE, value):
    #             self.append_command_log(f"Команда отправлена: регистр 0x{REG_SET_PRESSURE:02X}, значение 0x{value:04X}")
    #             self.append_command_log(f"Давление установлено: {value / 10} Pa")
    #
    #         # if was_stab:
    #         #     self.controller.write_register(REG_COMMAND, CMD_START)
    #     except ValueError:
    #         self.append_command_log("Ошибка: введите число")
    #
    # def _read_pressure(self):
    #     """Читает текущее значение уставки давления"""
    #     value = self.controller.read_register(REG_SET_PRESSURE)
    #     if value is not None:
    #         self.set_pressure_var.set(str(value / 10))
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_SET_PRESSURE:02X}, ответ 0x{value:04X}")
    #     else:
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_SET_PRESSURE:02X}, ответа НЕТ")
    #
    # def _set_position(self):
    #     """Устанавливает позицию заслонки"""
    #     value = self.position_var_set.get()
    #     if self.controller.write_register(REG_SET_POSITION, value):
    #         self.append_command_log(f"Позиция установлена: {value}")
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_SET_POSITION:02X}, ответ 0x{value:04X}")
    #     else:
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_SET_POSITION:02X}, ответа НЕТ")
    #
    # def _set_position_var(self, value):
    #     """Меняет значение переменной с текстом установленного положения заслонки"""
    #     value = self.position_var_set.get()
    #     self.position_text_var_set.set(f"Позиция уст.: {value}")
    #
    # def _set_middle_position(self):
    #     """Устанавливает среднее положение заслонки без блокировки главного цикла"""
    #     value = self.controller.read_register(REG_STATUS)
    #     if value is None:
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_STATUS:02X}, ответа НЕТ")
    #         return
    #     else:
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_STATUS:02X}, ответ 0x{value:04X}")
    #
    #     self.controller.write_register(REG_COMMAND, CMD_OPEN)
    #     self.append_command_log(f"Команда отправлена: регистр 0x{REG_COMMAND:02X}, значение  0x{CMD_OPEN:04X}")
    #     # Запускаем проверку каждые 1000 мс до наступления нужного статуса
    #     self.window.after(1000, self._check_and_set_middle)
    #
    # def _check_and_set_middle(self):
    #     """Проверяет, установлен ли нужный статус, и отправляет команду установки среднего положения"""
    #     status = self.controller.read_register(REG_STATUS)
    #     self.append_command_log(
    #         f"Команда отправлена: регистр 0x{REG_STATUS:02X}, "
    #         f"ответ 0x{status:04X}" if status is not None else
    #         f"Команда отправлена: регистр 0x{REG_STATUS:02X}, ответ None"
    #     )
    #     if status is not None and (status & 0x02):
    #         self.controller.write_register(REG_COMMAND, CMD_MIDDLE_POSITION)
    #         self.append_command_log(f"Команда отправлена: регистр 0x{REG_COMMAND:02X}, значение  0x{CMD_MIDDLE_POSITION:04X}")
    #     else:
    #         # Если условие не выполнено, проверяем снова через 1000 мс
    #         self.window.after(1000, self._check_and_set_middle)
    #
    # def _log_data(self):
    #     """Логирование данных через модуль DataLogger"""
    #     try:
    #         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #
    #         # Получаем текущие значения
    #         temp = self.temperature_var.get().replace(" °C", "") if self.temperature_var.get() != "---" else None
    #         pressure = self.measured_pressure_var.get().replace(" Pa", "") if self.measured_pressure_var.get() != "---" else None
    #         position = self.position_var.get()
    #
    #         # Получаем статус в виде битовой маски
    #         status = 0
    #         for i, (name, var) in enumerate(self.status_vars.items()):
    #             status |= int(var.get()) << i
    #
    #         # Передаем данные логгеру
    #         self.logger.add_data(current_time, temp, pressure, position, status)
    #
    #     except Exception as e:
    #         self.append_command_log(f"Ошибка при логировании данных: {e}")
    #
    # def _start_log(self, is_on):
    #     self.log_enable.set(is_on)
    #     if is_on:
    #         print(f"Логирование запущено")
    #     else:
    #         print(f"Логирование остановлено")
    #
    # def _rec_to_log(self, n):
    #     """Выполняет n измерений и сохраняет их в лог"""
    #     start_measurement_time = time.time()
    #     measurements = []
    #
    #     print(f"Старт {n} измерений")
    #     while len(measurements) < n and time.time() - start_measurement_time < 60:
    #         print(f"{len(measurements)} измерение")
    #         # Получаем текущие значения
    #         temp = self.controller.read_register(REG_TEMPERATURE)
    #         pressure = self.controller.read_register(REG_MEASURED_PRESSURE)
    #         address, value = self.controller.position_queue_LO.get()
    #         if address == REG_POSITION_LO:
    #             position_lo = value
    #             self.append_command_log(f'pos_lo: {position_lo}')
    #         address, value = self.controller.position_queue_HI.get()
    #         if address == REG_POSITION_HI:
    #             position_hi = value
    #             self.append_command_log(f'pos_hi: {position_hi}')
    #         position = (position_hi << 16) | position_lo
    #         status = self.controller.read_register(REG_STATUS)
    #
    #         if None in (temp, pressure, position, status):
    #             continue  # Пропускаем если нет данных
    #
    #         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #
    #         measurements.append([
    #             current_time,
    #             temp / 10.0,  # Температура
    #             pressure / 10.0,  # Давление
    #             position,  # Позиция
    #             status  # Статус
    #         ])
    #
    #         time.sleep(0.2)  # Интервал между измерениями
    #
    #     # Сохраняем все измерения одним пакетом
    #     if measurements:
    #         try:
    #             for measurement in measurements:
    #                 self.logger.add_data(*measurement)
    #             self.logger.flush()  # Принудительно сохраняем
    #             self.append_command_log(f"Успешно сохранено {len(measurements)} измерений")
    #         except Exception as e:
    #             self.append_command_log(f"Ошибка при сохранении измерений: {e}")

    # def append_command_log(self, message: str):
    #     """Добавляет строку в окно вывода команд"""
    #
    #     def _append():
    #         self.command_output.insert('end', message + '\n')
    #         self.command_output.see('end')
    #
    #     self.command_output.after(0, _append)
    #

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

# class GuiOutputRedirector:
#     def __init__(self, gui_instance):
#         self.gui = gui_instance
#
#     def write(self, message):
#         # Убираем пустые строки и перевод строки
#         if message.strip():
#             self.gui.append_command_log(message.strip())
#
#     def flush(self):
#         pass  # требуется для совместимости с sys.stdout