# zmx_echelle_editor.py
"""
Zmx Echelle Editor
Редактирует .zmx (Zemax Classic) файл: обновляет PRAM(10) (штрихи решетки),
первые 4 WAVE записи каждой конфигурации (центр, центр+delta, левый край, правый край),
наклон решетки (поверхность 10), наклоны Y для поверхностей 6,7,13,14 и фокус (поверхность 19).

Автор: помощник (адаптируем при запросе)
"""

from __future__ import annotations
import re
import math
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

from lib.echelle.dataClasses import Spectrometer, ConfigOGF, OrderEdges
from lib.echelle.echelleMath import (
    lambda_center_k, lambda_range,
    find_orders_range,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------- Парсинг/замена строк .zmx -----------------------
_re_pram = re.compile(r'^(PRAM)\s+(\d+)\s+(\d+)\s+([+-]?\d*\.?\d+[Ee][+-]?\d+)(.*)$', re.IGNORECASE)
_re_wave = re.compile(r'^(WAVE)\s+(\d+)\s+(\d+)\s+([+-]?\d*\.?\d+[Ee][+-]?\d+)(.*)$', re.IGNORECASE)
_re_surf = re.compile(r'^(SURF|SUR)\s+(\d+)(.*)$', re.IGNORECASE)
_re_any_number = re.compile(r'([+-]?\d*\.?\d+(?:[Ee][+-]?\d+)?)')


def _format_e(val: float) -> str:
    """Формат с экспонентой: совпадает с вашим .zmx стилем (15 значащих ~)"""
    return f"{val:.12E}"


class ZmxEchelleEditor:
    def __init__(
        self,
        filename: str,
        spectrometer: Spectrometer,
        config: ConfigOGF,
        center_config_idx: int = 1,
        output: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.filename = filename
        self.lines_per_mm = spectrometer.grating.lines_per_mm
        self.gamma_deg = math.degrees(spectrometer.grating.gamma_rad)
        self.gamma_rad = spectrometer.grating.gamma_rad
        self.lambda_min_nm = config.lambda_min
        self.lambda_max_nm = config.lambda_max
        self.resolution_nm = config.res_limit
        self.center_config_idx = int(center_config_idx)
        self.center_lambda = config.lambda_ctr
        self.prism_angle_deg = math.degrees(spectrometer.prism.wedge_angle_rad)
        self.new_focus_mm = spectrometer.focal_mm
        self.echellegramma_orders = spectrometer.orders_in_lambda
        self.dry_run = bool(dry_run)
        if output is None:
            self.output = filename.rsplit('.', 1)[0] + '-modified.zmx'
        else:
            self.output = output

        # file content
        self.lines: List[str] = []
        # discovered configs
        self.configs: List[int] = []

    def read(self):
        with open(self.filename, 'r', encoding='utf-16', errors='ignore') as f:
            text = f.read()
        # splitlines автоматически обработает \r\n
        self.lines = text.splitlines()
        logger.info("Loaded %d lines from %s", len(self.lines), self.filename)

    def find_configs(self):
        cfgs = set()
        for ln in self.lines:
            m = _re_wave.match(ln)
            if m:
                cfgs.add(int(m.group(3)))
        self.configs = sorted(cfgs)
        logger.info("Found configs: %s", self.configs)

    def compute_orders_and_wavelengths(self):
        """
        По параметрам схемы находим kmin/kmax и затем сопоставляем k -> config.
        Возвращаем dict: config -> dict(k, center_nm, lam_min_nm, lam_max_nm)
        """
        kmin = min(self.echellegramma_orders)
        kmax = max(self.echellegramma_orders)
        ks_list = list(range(kmin, kmax + 1))
        if kmin == 0:
            # fallback: попробуем грубо подобрать, используя lambda_center for k=1..50
            ks_possible = []
            for k in range(1, 200):
                lam_nm = lambda_center_k(self.lines_per_mm, self.gamma_rad, k) * 1e6
                if self.lambda_min_nm * 0.99 <= lam_nm <= self.lambda_max_nm * 1.01:
                    ks_possible.append(k)
            if ks_possible:
                kmin, kmax = min(ks_possible), max(ks_possible)
                ks_list = ks_possible
            else:
                # окончательный fallback — задаём kmin=1,kmax=5
                kmin, kmax = 1, 5
                ks_list = list(range(kmin, kmax + 1))
                logger.warning("Не удалось автоматически найти порядки в "
                               "диапазоне длин волн; использую fallback k=[1..5]")

        logger.info("Orders determined: kmin=%d kmax=%d (candidate list len=%d)", kmin, kmax, len(ks_list))

        # configs -> assign orders by linear distribution
        configs = self.configs
        if not configs:
            raise RuntimeError("Не найдены конфигурации WAVE в файле")

        mapping: Dict[int, Dict] = {}
        lambda1_nm = lambda_center_k(self.lines_per_mm, self.gamma_rad, 1)
        for k in range(kmin, kmax):
            lam_min_nm, lam_max_nm = self.echellegramma_orders[k] * 1e6
            if lam_min_nm < self.center_lambda < lam_max_nm:
                lam_cent_nm = lambda_center_k(self.lines_per_mm, self.gamma_rad, k)
                mapping[self.center_config_idx] = {
                    "k": k,
                    "center_nm": float(lam_cent_nm),
                    "lam_min_nm": float(lam_min_nm),
                    "lam_max_nm": float(lam_max_nm),
                }
        lam_min_nm, lam_max_nm = self.echellegramma_orders[kmax] * 1e6
        lam_cent_nm = lambda_center_k(self.lines_per_mm, self.gamma_rad, kmax)
        mapping[self.center_config_idx + 1] = {
            "k": kmax,
            "center_nm": float(lam_cent_nm),
            "lam_min_nm": float(lam_min_nm),
            "lam_max_nm": float(lam_max_nm),
        }
        configs = configs[2:len(configs)]
        C = len(configs)
        for i, cfg in enumerate(configs):
            if C == 1:
                k = kmin
            else:
                # распределяем равномерно по целым порядкам
                val = kmin + i * ((kmax - 1) - kmin) / (C - 1) if C > 1 else kmin
                k = int(round(val))
                if any(v["k"] == k for v in mapping.values()):
                    if k == kmin:
                        k += 1
                    else:
                        k -= 1

                if k < 1:
                    k = 1
            lam_cent_nm = lambda_center_k(self.lines_per_mm, self.gamma_rad, k)
            lam_min_nm, lam_max_nm = self.echellegramma_orders[k] * 1e6

            mapping[cfg] = {
                "k": k,
                "center_nm": float(lam_cent_nm),
                "lam_min_nm": float(lam_min_nm),
                "lam_max_nm": float(lam_max_nm),
            }

        logger.info("Computed mapping for configs (sample): %s", {c: mapping[c] for c in list(mapping)[:min(4, len(mapping))]})
        self._mapping = mapping

        return mapping

    def replace_surface_parm(self, surface_idx: int, parm_idx: int, new_value: float) -> int:
        """
        Находит блок поверхности SURF <surface_idx> и заменяет значение у PARM <parm_idx>.
        Пример:
            SURF 10
              TYPE DGRATING
              ...
              PARM 1 8.15E-2 <-- заменим это значение
              PARM 2 46E+1
        Аргументы:
            surface_idx : int  — номер поверхности (например 10)
            parm_idx    : int  — номер параметра внутри поверхности (например 2)
            new_value   : float — новое значение (будет записано в экспоненциальной форме)
        Возвращает:
            количество изменённых строк (0 если не найдено)
        """
        in_target_surface = False
        changed = 0
        new_lines = []
        formatted_value = _format_e(new_value)  # например 8.150000000000E+001

        surf_header = f"SURF {surface_idx}"
        parm_prefix = f"  PARM {parm_idx}"

        for ln in self.lines:
            # начало блока поверхности
            if ln.strip().startswith("SURF "):
                # если новая поверхность — проверяем, совпадает ли номер
                in_target_surface = ln.strip().lower() == surf_header.lower()

            # если мы находимся внутри целевой поверхности
            if in_target_surface and ln.strip().startswith(f"PARM {parm_idx}"):
                # пример строки: "  PARM 1 8.6E+1"
                parts = ln.strip().split()
                if len(parts) >= 3:
                    # заменим только значение
                    old_val = parts[2]
                    ln = f"  PARM {parm_idx} {formatted_value}"
                    changed += 1
                else:
                    logger.warning(f"Неверный формат строки PARM: {ln}")
                # примечание: Zemax допускает, что внутри блока несколько PARM с тем же индексом — изменяем только первое
                in_target_surface = False  # после замены можно выйти из блока

            new_lines.append(ln)

        self.lines = new_lines
        logger.info("Surface %d PARM %d updated %d times (new value = %s)", surface_idx, parm_idx, changed,
                    formatted_value)
        return changed

    def _replace_wave_entries(self):
        """
        Заменяет WAVE и PRAM строки на основе рассчитанных данных из self._mapping.

        Для каждой конфигурации из mapping:
          - WAVE:
              1 -> center_nm (в микрометрах)
              2 -> center_nm + resolution_nm (в микрометрах)
              3 -> lam_min_nm (в микрометрах)
              4 -> lam_max_nm (в микрометрах)
          - PRAM:
              заменяет строку "PRAM  10  <cfg> ..." — устанавливает новый порядок (k)
                в 5-м числовом поле (после значения длины волны)
        """
        mapping = getattr(self, "_mapping", None)
        if mapping is None:
            raise RuntimeError("Mapping not computed")

        # Подготовка таблицы замен
        replacements_wave: Dict[Tuple[int, int], float] = {}  # (idx, cfg) -> value (um)
        replacements_pram: Dict[int, Tuple[float, int]] = {}  # cfg -> (lambda_nm, k)

        for cfg, info in mapping.items():
            k = int(info["k"])
            center_mkm = info["center_nm"] / 1000.0
            lam_min_mkm = info["lam_min_nm"] / 1000.0
            lam_max_mkm = info["lam_max_nm"] / 1000.0
            res_mkm = self.resolution_nm / 1000.0

            raplace_center = cfg == self.center_config_idx or cfg == self.center_config_idx + 1

            replacements_wave[(1, cfg)] = center_mkm if raplace_center else lam_min_mkm
            replacements_wave[(2, cfg)] = (center_mkm + res_mkm) if cfg == self.center_config_idx else lam_max_mkm
            replacements_wave[(3, cfg)] = lam_min_mkm
            replacements_wave[(4, cfg)] = lam_max_mkm
            replacements_pram[cfg] = (center_mkm, k)

        # Проход по строкам и замена
        new_lines = []
        wave_changed = 0
        pram_changed = 0

        for ln in self.lines:
            # ---- Замена WAVE ----
            m_wave = _re_wave.match(ln)
            if m_wave:
                idx = int(m_wave.group(2))
                cfg = int(m_wave.group(3))
                rest = m_wave.group(5)
                key = (idx, cfg)
                if key in replacements_wave:
                    val = replacements_wave[key]
                    newln = f"{m_wave.group(1):<4s}{idx:>4d}{cfg:>4d} {val:.12E}{rest}"
                    new_lines.append(newln)
                    wave_changed += 1
                    continue

            # ---- Замена PRAM 10 ----
            if ln.strip().startswith("PRAM  10"):
                parts = ln.split()
                if len(parts) >= 5:
                    cfg = int(parts[2])
                    if cfg in replacements_pram:
                        lam_nm, k = replacements_pram[cfg]
                        # parts[3] — порядок (по образцу ваших строк)
                        parts[3] = f"{k:.12E}"
                        ln = " ".join(parts)
                        pram_changed += 1

            new_lines.append(ln)

        self.lines = new_lines
        logger.info(
            "Replaced %d WAVE entries and %d PRAM entries (updated wavelengths and orders)",
            wave_changed, pram_changed
        )
        return wave_changed + pram_changed

    def _update_surface_param_by_line_prefix(
            self,
            surf_idx: int,
            param_name_candidates: List[str],
            new_value: float,
            sign_flip: bool = False
    ) -> int:
        """
        Универсальная функция: ищет строку, содержащую SURF <surf_idx> и одно из param_name_candidates,
        затем заменяет первое число в строке (после имени), либо заменяет все числовые поля в строке — осторожно.
        Возвращает число изменённых строк.
        Если sign_flip=True, new_value записывается как +/- попеременно при вызове внешне.
        """
        changed = 0
        out = []
        for ln in self.lines:
            m = _re_surf.match(ln)
            if m and int(m.group(2)) == surf_idx:
                # если строка содержит явно текст param_name рядом —
                # попробуем заменить первое число в строке после названия
                lower = ln.lower()
                found = False
                for key in param_name_candidates:
                    if key.lower() in lower:
                        found = True
                        break
                if found:
                    # заменим первое числовое в строке на формат E
                    def repl_num(match):
                        nonlocal changed
                        changed += 1
                        val = new_value
                        if sign_flip:
                            val = -new_value
                        return _format_e(val)
                    new_ln = _re_any_number.sub(repl_num, ln, count=1)
                    out.append(new_ln)
                    continue
            out.append(ln)
        self.lines = out
        return changed

    def _replace_surface_param_value(
            self,
            surface_idx: int,
            param_name: str,
            sub_idx: Optional[int],
            value_pos: int,
            new_value: float
    ) -> int:
        """
        Универсально изменяет конкретное значение внутри строки параметра поверхности Zemax.

        Пример:
            SURF 10
              SCBD 1 0 0 0.000000000000E+000 0.000000000000E+000 3.900000000000E+001 ...
            -> заменить 5-е значение у SCBD 1

        Аргументы:
            surface_idx : int  — номер поверхности (например 10)
            param_name  : str  — имя параметра (например "SCBD")
            sub_idx     : int  — индекс подблока (например 1)
            value_pos   : int  — позиция значения для изменения (нумерация с 1)
            new_value   : float — новое значение (будет записано в экспоненциальной форме)

        Возвращает:
            количество изменённых строк (0, если не найдено)
        """
        changed = 0
        in_target_surface = False
        new_lines = []
        formatted_value = _format_e(new_value)
        surf_header = f"SURF {surface_idx}"
        if sub_idx is None:
            param_prefix = f"{param_name}"
            shift = 0
        else:
            param_prefix = f"{param_name} {sub_idx}"
            shift = 1

        for ln in self.lines:
            # Проверяем, начался ли новый блок поверхности
            if ln.strip().startswith("SURF "):
                # Входим в нужный блок, если номер совпадает
                in_target_surface = ln.strip().upper() == surf_header.upper()

            # Если мы внутри нужной поверхности — проверяем строки параметров
            if in_target_surface and ln.strip().startswith(param_prefix):
                parts = ln.strip().split()
                # parts[0] = param_name, parts[1] = sub_idx, далее числовые значения
                if len(parts) - shift  > value_pos:
                    old_val = parts[value_pos + shift]
                    parts[value_pos + shift] = formatted_value
                    ln = "  " + " ".join(parts)
                    changed += 1
                    logger.debug(
                        f"Surface {surface_idx}: {param_name} {sub_idx}, value#{value_pos} "
                        f"changed {old_val} -> {formatted_value}"
                    )
                else:
                    logger.warning(
                        f"Surface {surface_idx}: {param_name} {sub_idx} has fewer than {value_pos} values ({ln})"
                    )

            new_lines.append(ln)

        self.lines = new_lines
        logger.info(
            "Surface %d %s %d: updated %d lines (value #%d = %s)",
            surface_idx, param_name, 0 if sub_idx is None else sub_idx, changed, value_pos, formatted_value
        )
        return changed

    def _update_prism_surfaces_y(self, prism_angle_deg: float):
        """
        Для поверхностей 6,7,13,14 по Y: угол задаётся как +/- (prism_angle/2).
        Соответие знаков: 6 -> -, 7 -> +, 13 -> +, 14 -> -
        Будем искать строки, начинающиеся с SURF <N> и заменять первое встречное число.
        """
        mapping = {6: -prism_angle_deg/2.0, 7: prism_angle_deg/2.0, 13: prism_angle_deg/2.0, 14: -prism_angle_deg/2.0}
        total_changed = 0
        for surf, val in mapping.items():
            if self._replace_surface_param_value(surf, "SCBD", 1, 6, val):
                logger.info("Surface %d: replaced fields with %f deg (Y tilt)", surf, val)
                total_changed += 1
        return total_changed

    def _update_focus_surface19(self, new_focus_mm: float):
        """
        Находим SURF 19 и заменяем первое числовое поле (толщина/заданное значение) на new_focus_mm - 12.5
        """
        target = new_focus_mm - 12.5
        changed = 0
        out = []
        for ln in self.lines:
            m = _re_surf.match(ln)
            if m and int(m.group(2)) == 19:
                def repl(match):
                    nonlocal changed
                    changed += 1
                    return _format_e(target)
                newln = _re_any_number.sub(repl, ln, count=1)
                out.append(newln)
                continue
            out.append(ln)
        self.lines = out
        logger.info("Surface 19: updated %d lines, set f-12.5 = %f", changed, target)
        return changed

    def write(self):
        if self.dry_run:
            logger.info("Dry run - not writing output.")
            return
        with open(self.output, 'w', encoding='utf-16', newline='\r\n') as f:
            for line in self.lines:
                f.write(line + '\r\n')
        logger.info("Saved Zemax file: %s (UTF-16 LE with BOM)", self.output)

    def process(self):
        # high-level
        self.read()
        self.find_configs()
        self.compute_orders_and_wavelengths()
        self._replace_surface_param_value(10, "PARM", 1, 1, self.lines_per_mm / 1000)
        # self.replace_surface_parm(10, 1, self.lines_per_mm / 1000)
        self._replace_wave_entries()
        # grating tilt (surface 10) — используем gamma as baseline tilt X (user said примерно -7 deg)
        self._replace_surface_param_value(10, "SCBD", 1, 5, self.gamma_deg - 5.5)
        # prism surfaces Y
        self._update_prism_surfaces_y(self.prism_angle_deg)
        # focus on surface 19
        # if self.new_focus_mm is not None:
        #     self._replace_surface_param_value(19, "DISZ", None, 1, - self.new_focus_mm + 12.5)
        # write result
        self.write()
        logger.info("Processing completed. Mapping summary: %s", getattr(self, "_mapping", {}))
