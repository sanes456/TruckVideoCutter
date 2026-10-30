"""
Truck Video Cutter - Автоматический поиск, скачивание, нарезка и фильтрация видео грузовиков.
Интерфейс: CustomTkinter (Тёмная тема, русский язык).
Поддержка: 
- Глубокий поиск по всему YouTube видео с грузовиками ВИД СВЕРХУ / ДРОН (Top-Down / Aerial)
- Автоматический ИИ-контроль водяных знаков и титров: адаптивная обрезка краевых логотипов и пропуск видео с неустранимыми центральными надписями
- База истории: исключение повторно скачанных видео (всегда только новые ролики)
- Тематики грузоперевозок (фуры с грузом, тяжеловозы с негабаритом, самосвалы, бездорожье)
- Длинные видео и YouTube Shorts
- Специализированный ИИ-анализ YOLOv8 на грузовики с грузом/прицепом
"""

import os
import sys
import re
import time
import json
import shutil
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import requests
import yt_dlp
from ultralytics import YOLO

# Настройка темы CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Путь к рабочей директории, файлу истории, пресету настроек и памяти самообучения ИИ
PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_FILE = PROJECT_ROOT / "downloaded_history.json"
SETTINGS_FILE = PROJECT_ROOT / "settings_preset.json"
AI_MEMORY_FILE = PROJECT_ROOT / "ai_learning_memory.json"

# Заводские настройки по умолчанию для пресетов
DEFAULT_SETTINGS: Dict[str, Any] = {
    "source_mode": "Автопоиск YouTube",
    "view_angle": "Только вид сверху / дрон (Top-Down / Aerial)",
    "cargo_theme": "Все виды техники (Автосортировка: грузовики / наземная / воздушная)",
    "youtube_type": "Все форматы (Длинные видео и Shorts)",
    "query": "",
    "pexels_key": "",
    "pixabay_key": "",
    "quality": "1080p Full HD (Стабильно, быстро)",
    "auth_mode": "Авто (файл cookies.txt при наличии)",
    "custom_cookie_path": None,
    "date_filter": "За всё время (без ограничений)",
    "min_duration": "0.1",
    "max_duration": "60",
    "format_mode": "Вертикальное 9:16 (1080x1920 - Shorts/Reels/TikTok)",
    "clip_count": "5",
    "clip_duration": "60",
    "output_dir": str((PROJECT_ROOT / "output_videos")),
    "watermark_mode": "Строгий фильтр (Только 100% чистые видео без знаков)",
    "crop_slider": 0,
    "ai_filter": True,
    "ai_mode": "Грузовики, военная техника и дроны (Автосортировка: грузовики/наземная/воздушная)",
    "ai_learning": True,
    "avoid_repeats": True,
    "remove_audio": True,
}


def ensure_netscape_cookies(filepath: str) -> Optional[str]:
    """
    Проверяет файл cookies на соответствие формату Netscape.
    Если файл в формате HTTP Header (LOGIN_INFO=...; PREF=...) или JSON,
    автоматически конвертирует его в валидный формат Netscape для yt-dlp.
    """
    if not filepath or not os.path.isfile(filepath) or os.path.getsize(filepath) < 10:
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        
        # 1. Проверяем, является ли файл уже валидным Netscape форматом
        lines = content.splitlines()
        is_netscape = False
        if any(l.startswith("# Netscape HTTP Cookie File") or l.startswith("# HTTP Cookie File") for l in lines[:10]):
            is_netscape = True
        else:
            for l in lines:
                l = l.strip()
                if l and not l.startswith("#") and l.count("\t") >= 6:
                    is_netscape = True
                    break
        
        if is_netscape:
            return str(Path(filepath).resolve())

        # 2. Если не Netscape, конвертируем
        expiry = int(time.time()) + 3600 * 24 * 365 # 1 год
        netscape_lines = [
            "# Netscape HTTP Cookie File",
            "# http://curl.haxx.se/rfc/cookie_spec.html",
            "# This file was automatically converted to Netscape format",
            ""
        ]

        # Вариант А: JSON формат
        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "cookies" in data:
                    data = data["cookies"]
                if isinstance(data, list):
                    for item in data:
                        domain = item.get("domain", ".youtube.com")
                        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
                        path = item.get("path", "/")
                        secure = "TRUE" if item.get("secure", True) else "FALSE"
                        exp = int(item.get("expirationDate") or item.get("expires") or expiry)
                        name = item.get("name", "")
                        value = item.get("value", "")
                        if name:
                            netscape_lines.append(f"{domain}\t{include_sub}\t{path}\t{secure}\t{exp}\t{name}\t{value}")
            except Exception:
                pass

        # Вариант Б: HTTP Header формат (LOGIN_INFO=...; PREF=...; ...)
        if len(netscape_lines) <= 4:
            pairs = content.split(";")
            for p in pairs:
                p = p.strip()
                if not p or "=" not in p:
                    continue
                parts = p.split("=", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if k and v:
                    domain = ".youtube.com"
                    netscape_lines.append(f"{domain}\tTRUE\t/\tTRUE\t{expiry}\t{k}\t{v}")

        # Если успешно сконвертировали хотя бы 1 cookie
        if len(netscape_lines) > 4:
            converted_content = "\n".join(netscape_lines) + "\n"
            converted_path = PROJECT_ROOT / "youtube_cookies_netscape.txt"
            with open(converted_path, "w", encoding="utf-8") as f:
                f.write(converted_content)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(converted_content)
            except Exception:
                pass
            return str(converted_path.resolve())

        return None
    except Exception:
        return None


def find_active_cookie_file(custom_path: Optional[str] = None) -> Optional[str]:
    """Поиск и валидация доступного файла cookies для YouTube в формате Netscape."""
    if custom_path:
        valid_custom = ensure_netscape_cookies(custom_path)
        if valid_custom:
            return valid_custom
    
    # Кандидаты в папке проекта и Downloads
    search_paths = [
        PROJECT_ROOT / "youtube_cookies_netscape.txt",
        PROJECT_ROOT / "cookies.txt",
        PROJECT_ROOT / "youtube_cookies.txt",
        PROJECT_ROOT / "youtube.com_cookies.txt",
        Path.home() / "Downloads" / "www.youtube.com_cookies.txt",
        Path.home() / "Downloads" / "cookies.txt",
        Path.home() / "Downloads" / "youtube_cookies.txt"
    ]
    for p in search_paths:
        if p.is_file() and p.stat().st_size > 10:
            valid = ensure_netscape_cookies(str(p))
            if valid:
                return valid
    return None


def load_download_history() -> Dict[str, Any]:
    """Загрузка истории ранее скачанных видео."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_download_history_entry(video_id: str, title: str):
    """Сохранение ID видео в историю во избежание повторов."""
    hist = load_download_history()
    hist[video_id] = {
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clear_download_history_file():
    """Очистка базы истории скачанных видео."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# Список служебных стоп-слов для очистки текста при извлечении ключевых паттернов
NLP_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "4k", "hd", "uhd", "video", "shorts", "1080p", "60fps", "full", "new",
    "top", "view", "drone", "дрон", "вид", "видео", "клип", "4к", "хд",
    "short", "watch", "online", "free", "best", "super", "real", "life",
    "this", "that", "these", "those", "how", "what", "where", "when", "why"
}


class AILearningEngine:
    """
    Интеллектуальная система самообучения на скачиваемых видео:
    - Адаптивное ранжирование и оптимизация поисковых запросов по их результативности
    - Оценка надежности и качества YouTube-каналов (Channel Authority Scoring)
    - Обучение на названиях и тегах успешных видео (извлечение ключевых паттернов и n-грамм)
    - Автоматическая генерация новых 'умных' поисковых запросов (Smart Query Synthesis)
    - Префильтрация кандидатов перед скачиванием на основе накопленного опыта
    """
    def __init__(self, memory_file: Path = AI_MEMORY_FILE):
        self.memory_file = memory_file
        self.lock = threading.Lock()
        self.memory: Dict[str, Any] = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        if self.memory_file.is_file():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "query_stats" in data:
                    return data
            except Exception:
                pass
        return {
            "version": "1.0",
            "total_analyzed_videos": 0,
            "total_successful_clips": 0,
            "query_stats": {},          # query -> {"uses": int, "success": int, "fails": int, "clips": int, "score": float}
            "channel_stats": {},        # channel_name -> {"success": int, "fails": int, "clips": int, "rating": float}
            "positive_keywords": {},    # word/ngram -> weight (float)
            "negative_keywords": {},    # word -> count (int)
            "smart_queries": []         # list of str
        }

    def _save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _extract_tokens(self, text: str) -> List[str]:
        cleaned = re.sub(r'[^\w\s-]', ' ', text.lower())
        tokens = [w.strip() for w in cleaned.split() if len(w.strip()) >= 3 and w.strip() not in NLP_STOP_WORDS]
        return tokens

    def _extract_ngrams(self, text: str) -> List[str]:
        tokens = self._extract_tokens(text)
        ngrams = list(tokens)
        for i in range(len(tokens) - 1):
            pair = f"{tokens[i]} {tokens[i+1]}"
            ngrams.append(pair)
        return ngrams

    def record_success(self, query: str, video_title: str, channel: str = "", tags: Optional[List[str]] = None, clips_count: int = 1) -> List[str]:
        """Фиксация успешного скачивания и нарезки: обучение на запросе, канале и ключевых словах."""
        new_queries = []
        with self.lock:
            self.memory["total_analyzed_videos"] += 1
            self.memory["total_successful_clips"] += clips_count

            # 1. Рейтинг поискового запроса
            q_stats = self.memory["query_stats"].setdefault(query, {
                "uses": 0, "success": 0, "fails": 0, "clips": 0, "score": 1.0
            })
            q_stats["uses"] += 1
            q_stats["success"] += 1
            q_stats["clips"] += clips_count
            q_stats["score"] = round((q_stats["clips"] * 2.5 + q_stats["success"] * 2.0) / max(1, q_stats["uses"]), 2)

            # 2. Рейтинг канала
            if channel:
                c_stats = self.memory["channel_stats"].setdefault(channel, {
                    "success": 0, "fails": 0, "clips": 0, "rating": 1.0
                })
                c_stats["success"] += 1
                c_stats["clips"] += clips_count
                c_stats["rating"] = round(min(10.0, 1.0 + c_stats["clips"] * 0.75), 2)

            # 3. Извлечение позитивных токенов и n-грамм
            combined_text = f"{video_title} {' '.join(tags or [])}"
            ngrams = self._extract_ngrams(combined_text)
            for ng in ngrams:
                cur_w = self.memory["positive_keywords"].get(ng, 0.0)
                self.memory["positive_keywords"][ng] = round(cur_w + (0.5 * clips_count), 2)

            # 4. Синтез умных поисковых запросов на основе лучших ключевых фраз
            sorted_kw = sorted(self.memory["positive_keywords"].items(), key=lambda x: x[1], reverse=True)
            top_kw = [k for k, v in sorted_kw[:12] if len(k.split()) >= 2]
            
            angles = ["drone 4k top down", "aerial view 4k", "вид сверху 4k"]
            for kw in top_kw[:4]:
                for ang in angles:
                    candidate_q = f"{kw} {ang}".strip()
                    if candidate_q not in self.memory["smart_queries"] and candidate_q not in self.memory["query_stats"]:
                        self.memory["smart_queries"].append(candidate_q)
                        new_queries.append(candidate_q)
                        if len(self.memory["smart_queries"]) > 30:
                            self.memory["smart_queries"].pop(0)

            self._save_memory()
        return new_queries

    def record_failure(self, query: str, video_title: str, channel: str = "", reason: str = ""):
        """Обучение на браке (водяные знаки, игры, пустые кадры, отклонено ИИ)."""
        with self.lock:
            self.memory["total_analyzed_videos"] += 1

            # Снижаем рейтинг запроса
            if query in self.memory["query_stats"]:
                q_stats = self.memory["query_stats"][query]
                q_stats["uses"] += 1
                q_stats["fails"] += 1
                q_stats["score"] = round(max(0.1, q_stats["score"] * 0.85), 2)

            # Снижаем рейтинг канала при водяных знаках или браке
            if channel:
                c_stats = self.memory["channel_stats"].setdefault(channel, {
                    "success": 0, "fails": 0, "clips": 0, "rating": 0.0
                })
                c_stats["fails"] += 1
                c_stats["rating"] = round(max(-5.0, c_stats["rating"] - 1.2), 2)

            # Если отклонено из-за водяных знаков или игры - фиксируем стоп-токены
            if "водян" in reason.lower() or "игра" in reason.lower() or "не обнаружено" in reason.lower():
                tokens = self._extract_tokens(video_title)
                for t in tokens:
                    self.memory["negative_keywords"][t] = self.memory["negative_keywords"].get(t, 0) + 1

            self._save_memory()

    def rank_and_augment_queries(self, base_queries: List[str], theme_name: str = "", limit: int = 35) -> List[str]:
        """
        Ранжирование пула запросов с учетом накопленного опыта:
        - Запросы с высоким рейтингом идут первыми
        - Внедрение автоматически синтезированных умных запросов
        - Запросы по топовым проверенным каналам
        """
        with self.lock:
            def get_score(q: str) -> float:
                if q in self.memory["query_stats"]:
                    return self.memory["query_stats"][q].get("score", 1.0)
                return 1.0

            sorted_base = sorted(base_queries, key=get_score, reverse=True)

            augmented: List[str] = []

            # 1. Сначала ставим топовые умные запросы из памяти обучения
            smart_q = list(self.memory.get("smart_queries", []))
            for sq in smart_q[-6:]:
                if sq not in augmented:
                    augmented.append(sq)

            # 2. Добавляем запросы по авторитетным проверенным каналам (рейтинг >= 3.0)
            top_channels = [c for c, data in self.memory["channel_stats"].items() if data.get("rating", 0) >= 3.0]
            for ch in top_channels[:3]:
                ch_query = f"{ch} drone 4k"
                if ch_query not in augmented and ch_query not in sorted_base:
                    augmented.append(ch_query)

            # 3. Добавляем отсортированные базовые запросы
            for bq in sorted_base:
                if bq not in augmented:
                    augmented.append(bq)

            return augmented[:limit]

    def score_candidate_entry(self, title: str, channel: str = "", description: str = "") -> float:
        """
        Предварительная оценка кандидата перед скачиванием:
        Позволяет скачивать самые перспективные видео в первую очередь.
        """
        score = 0.0
        with self.lock:
            if channel and channel in self.memory["channel_stats"]:
                ch_rating = self.memory["channel_stats"][channel].get("rating", 0.0)
                score += ch_rating * 2.0

            text = f"{title} {description}".lower()
            for kw, weight in self.memory["positive_keywords"].items():
                if kw in text:
                    score += min(5.0, weight * 0.4)

            for nkw, count in self.memory["negative_keywords"].items():
                if nkw in text:
                    score -= min(5.0, count * 0.5)

        return score

    def get_stats_summary(self) -> Dict[str, Any]:
        """Сводка данных обучения для отображения в интерфейсе."""
        with self.lock:
            top_q = sorted(self.memory["query_stats"].items(), key=lambda x: x[1].get("score", 0), reverse=True)[:5]
            top_c = sorted(self.memory["channel_stats"].items(), key=lambda x: x[1].get("rating", 0), reverse=True)[:5]
            top_kw = sorted(self.memory["positive_keywords"].items(), key=lambda x: x[1], reverse=True)[:8]

            return {
                "total_analyzed": self.memory["total_analyzed_videos"],
                "total_clips": self.memory["total_successful_clips"],
                "total_queries_tracked": len(self.memory["query_stats"]),
                "total_channels_rated": len(self.memory["channel_stats"]),
                "smart_queries_count": len(self.memory.get("smart_queries", [])),
                "top_queries": [(q, d["score"], d["clips"]) for q, d in top_q],
                "top_channels": [(c, d["rating"], d["clips"]) for c, d in top_c if d["rating"] > 0],
                "top_keywords": top_kw,
                "smart_queries": list(self.memory.get("smart_queries", []))[-5:]
            }

    def reset_memory(self):
        """Сброс накопленной базы обучения."""
        with self.lock:
            self.memory = {
                "version": "1.0",
                "total_analyzed_videos": 0,
                "total_successful_clips": 0,
                "query_stats": {},
                "channel_stats": {},
                "positive_keywords": {},
                "negative_keywords": {},
                "smart_queries": []
            }
            self._save_memory()


# Запрещенные ключевые слова (игры, анимации, симуляторы)
STRICT_BLACKLIST = {
    "game", "games", "gaming", "gameplay", "gta", "ets2", "ets 2", "euro truck",
    "beamng", "beam ng", "simulation", "simulator", "3d", "render", "animation",
    "cartoon", "carmod", "mod", "igra", "симулятор", "игра", "геймплей", "анимация",
    "мод", "мультфильм", "blender", "unreal engine", "unity",
    "arma", "arma 3", "arma3", "war thunder", "warthunder", "world of tanks",
    "wot", "dcs", "dcs world", "flight simulator", "msfs"
}

# Ключевые слова стоковых превью и водяных знаков
WATERMARK_META_BLACKLIST = {
    "shutterstock", "pond5", "getty images", "gettyimages", "getty",
    "istock", "istockphoto", "envato", "storyblocks", "artgrid",
    "watermark", "watermarked", "preview only", "preview_only",
    "stock footage preview", "stock video preview", "adobe stock preview",
    "alamy"
}

# Маркеры классификации техники: Воздушная (дроны, БПЛА) vs Грузовики vs Военная наземная
AIR_STRONG_KEYWORDS = {
    "uav", "бпла", "fpv", "квадрокоптер", "quadcopter", "multicopter",
    "hexacopter", "octocopter", "беспилотник", "беспилотники", "беспилотный",
    "беспилотная", "байрактар", "bayraktar", "ланцет", "lancet", "орлан",
    "orlan", "шахед", "shahed", "герань", "geran", "zala", "зала",
    "mavic", "мавик", "matrice", "баба яга", "baba yaga", "камикадзе",
    "kamikaze", "drone drop", "drone strike", "switchblade", "scan eagle",
    "mq-9", "reaper", "predator", "uav surveillance", "tactical uav",
    "reconnaissance drone", "surveillance drone", "fpv drone", "military drone",
    "военный дрон", "дрон бпла", "дрон разведчик", "ударный дрон", "fpv камикадзе"
}

AIR_SOFT_KEYWORDS = {
    "drone", "drones", "дрон", "дроны", "аэросъемка", "aerial footage",
    "birds eye", "overhead"
}

TRUCK_KEYWORDS = {
    "truck", "trucks", "semi", "semi truck", "semi trucks", "trailer", "trailers",
    "lorry", "lorries", "haul", "hauling", "heavy haul", "lowboy", "dump truck",
    "dump trucks", "tipper", "tippers", "man truck", "scania", "volvo truck",
    "freightliner", "peterbilt", "kenworth", "mack truck", "logging truck",
    "logging trucks", "timber truck", "road train", "big rig", "box truck",
    "flatbed", "tanker truck", "cement mixer", "cargo truck", "freight truck",
    "oversize load", "heavy transport",
    "грузовик", "грузовики", "грузовой", "грузовая", "грузоперевозки", "фура",
    "фуры", "тягач", "тягачи", "самосвал", "самосвалы", "лесовоз", "лесовозы",
    "тяжеловоз", "тяжеловозы", "негабарит", "негабаритный", "автопоезд",
    "прицеп", "полуприцеп", "дальнобой", "дальнобойщик", "дальнобойщики",
    "зерновоз", "мусоровоз", "бетономешалка", "топливозаправщик",
    "перевозка щебня", "перевозка песка", "перевозка леса", "бревна"
}

MILITARY_GROUND_STRONG_KEYWORDS = {
    "tank", "tanks", "armored", "armour", "btr", "bmp", "ifv", "apc", "mrap",
    "humvee", "tigr", "typhoon", "convoy", "military vehicle", "military vehicles",
    "military convoy", "military truck", "military transport", "artillery",
    "howitzer", "mlrs", "himars", "main battle tank", "panzer", "infantry vehicle",
    "танк", "танки", "бтр", "бмп", "брдм", "мтлб", "рсзо", "град", "торнадо",
    "ураган", "смерч", "сау", "артиллерия", "гаубица", "бронетранспортер",
    "броневик", "бронемашина", "колонна техники", "военная колонна",
    "военная техника", "военный грузовик", "армейский грузовик", "пехота",
    "леопард", "абрамс", "т-72", "т-90", "т-80", "т-62", "т-64", "бредли", "м113", "m113",
    "leopard", "abrams", "bradley", "challenger"
}

TRUCK_AND_GROUND_BRANDS = {
    "kamaz", "ural", "kraz", "маз", "зил", "газ 66", "камаз", "урал", "краз",
    "actros", "daf", "iveco", "scania", "volvo", "man"
}

GROUND_STRONG_KEYWORDS = TRUCK_KEYWORDS | MILITARY_GROUND_STRONG_KEYWORDS

# Пул поисковых запросов для поиска по всему YouTube с акцентом на ВИД СВЕРХУ / ДРОН (Top-Down)
TOP_DOWN_GLOBAL_QUERIES = [
    "top down semi trucks highway drone 4k",
    "bird eye view road traffic trucks 4k",
    "overhead drone freight trucks interstate 4k",
    "aerial view trucks highway 4k real life",
    "top down motorway traffic trucks drone",
    "overhead heavy haul oversize load truck 4k",
    "birds eye view cargo trucks highway 4k",
    "top down dump trucks dirt road drone 4k",
    "aerial view road train trucks drone 4k",
    "вид сверху фуры трасса дрон 4k",
    "вид сверху грузовики шоссе аэросъемка 4k",
    "overhead semi trucks interchange drone 4k",
    "drone top down commercial trucks 4k",
    "top down logging trucks highway drone 4k",
    "bird eye view heavy loaded trucks 4k"
]

TOP_DOWN_SHORTS_GLOBAL_QUERIES = [
    "top down truck drone #shorts",
    "overhead semi truck #shorts",
    "birds eye view trucks #shorts",
    "aerial view truck highway #shorts",
    "вид сверху фура дрон #shorts",
    "drone top down trucks #shorts",
    "overhead truck traffic #shorts",
    "birds eye semi truck #shorts",
    "top down kamaz drone #shorts",
    "drone view heavy truck #shorts"
]

# Пул поисковых запросов для наземной военной техники (танки, БМП, военные грузовики, колонны)
MILITARY_GROUND_LONG_QUERIES = [
    "military vehicles convoy drone 4k",
    "tanks armored vehicles top down drone 4k",
    "military trucks convoy aerial view 4k",
    "heavy military transport road train drone 4k",
    "armored fighting vehicles convoy top down 4k",
    "военная техника колонна вид сверху 4k",
    "танки бмп военные грузовики аэросъемка 4k",
    "армейские грузовики урал камаз колонна 4k",
    "military tank convoy highway drone 4k",
    "mlrs rocket artillery convoy drone 4k",
    "infantry fighting vehicle convoy aerial 4k",
    "военная колонна бронетехники вид сверху 4k"
]

MILITARY_GROUND_SHORTS_QUERIES = [
    "military convoy drone #shorts",
    "tanks military vehicles top down #shorts",
    "военная техника колонна дрон #shorts",
    "танки бмп вид сверху #shorts",
    "военный грузовик урал камаз дрон #shorts",
    "armored vehicles convoy #shorts",
    "tank column top down #shorts",
    "военный тягач танковоз #shorts"
]

# Пул поисковых запросов для воздушной техники (дроны, БПЛА, квадрокоптеры, FPV)
MILITARY_AIR_LONG_QUERIES = [
    "military uav surveillance drone 4k",
    "fpv drone heavy payload tactical 4k",
    "tactical uav reconnaissance drone aerial 4k",
    "military drone flight top down 4k",
    "военный дрон бпла аэросъемка 4k",
    "дрон бпла разведчик вид сверху 4k",
    "quadcopter drone uav aerial view 4k",
    "drone dropping payload test 4k",
    "fpv kamikaze drone flight 4k",
    "беспилотник бпла аэросъемка 4k",
    "surveillance uav drone flight 4k",
    "heavy lift quadcopter drone 4k"
]

MILITARY_AIR_SHORTS_QUERIES = [
    "fpv drone military #shorts",
    "military uav drone footage #shorts",
    "военный бпла дрон #shorts",
    "quadcopter drone #shorts",
    "fpv drone flight #shorts",
    "дрон камикадзе бпла #shorts",
    "drone drop payload #shorts",
    "uav surveillance #shorts"
]

# Смешанный пул: Военная наземная техника + Дроны и БПЛА
MILITARY_AND_DRONES_GLOBAL_QUERIES = [
    "military vehicles convoy drone 4k",
    "military uav surveillance drone 4k",
    "tanks armored vehicles top down drone 4k",
    "fpv drone heavy payload tactical 4k",
    "военная техника колонна вид сверху 4k",
    "военный дрон бпла аэросъемка 4k",
    "military trucks convoy aerial view 4k",
    "tactical uav reconnaissance drone aerial 4k",
    "танки бмп военные грузовики аэросъемка 4k",
    "quadcopter drone uav aerial view 4k",
    "armored fighting vehicles convoy top down 4k",
    "дрон бпла разведчик вид сверху 4k",
    "армейские грузовики урал камаз колонна 4k",
    "fpv kamikaze drone flight 4k"
]

MILITARY_AND_DRONES_SHORTS_QUERIES = [
    "military convoy drone #shorts",
    "fpv drone military #shorts",
    "tanks military vehicles top down #shorts",
    "военный бпла дрон #shorts",
    "военная техника колонна дрон #shorts",
    "military uav drone footage #shorts",
    "танки бмп вид сверху #shorts",
    "дрон камикадзе бпла #shorts",
    "военный грузовик урал камаз дрон #shorts",
    "quadcopter drone #shorts"
]

# Тематические поисковые запросы для грузовиков и техники (по категориям)
CARGO_THEMES: Dict[str, Dict[str, List[str]]] = {
    "Все виды техники (Автосортировка: грузовики / наземная / воздушная)": {
        "long": TOP_DOWN_GLOBAL_QUERIES + MILITARY_AND_DRONES_GLOBAL_QUERIES,
        "shorts": TOP_DOWN_SHORTS_GLOBAL_QUERIES + MILITARY_AND_DRONES_SHORTS_QUERIES
    },
    "Военная техника + Дроны и БПЛА (Автосортировка: наземная / воздушная)": {
        "long": MILITARY_AND_DRONES_GLOBAL_QUERIES,
        "shorts": MILITARY_AND_DRONES_SHORTS_QUERIES
    },
    "Военная наземная техника (Танки, БТР, Военные грузовики, Колонны)": {
        "long": MILITARY_GROUND_LONG_QUERIES,
        "shorts": MILITARY_GROUND_SHORTS_QUERIES
    },
    "Военные дроны и БПЛА (FPV, Квадрокоптеры, Разведчики, Ударные БПЛА)": {
        "long": MILITARY_AIR_LONG_QUERIES,
        "shorts": MILITARY_AIR_SHORTS_QUERIES
    },
    "Все виды (Грузовики, Военная техника, Дроны и БПЛА)": {
        "long": TOP_DOWN_GLOBAL_QUERIES + MILITARY_AND_DRONES_GLOBAL_QUERIES,
        "shorts": TOP_DOWN_SHORTS_GLOBAL_QUERIES + MILITARY_AND_DRONES_SHORTS_QUERIES
    },
    "Фуры с грузом и автопоезда (Semi Trucks / Cargo Trailers)": {
        "long": [
            "top down semi trucks loaded highway drone 4k",
            "aerial view cargo freight trucks highway 4k",
            "overhead semi trucks with trailer drone 4k",
            "loaded semi trucks birds eye view drone 4k",
            "lorry cargo transport top down drone 4k",
            "вид сверху фуры с грузом трасса дрон 4k"
        ],
        "shorts": [
            "top down semi truck with trailer #shorts",
            "overhead loaded cargo truck drone #shorts",
            "birds eye view semi truck highway #shorts",
            "вид сверху фура дрон #shorts"
        ]
    },
    "Тяжеловозы и негабаритный груз (Heavy Haul / Oversize / Machinery)": {
        "long": [
            "top down heavy haul oversize load truck drone 4k",
            "overhead oversize load truck carrying excavator 4k",
            "heavy transport road train oversize cargo drone 4k",
            "birds eye view lowboy truck heavy machinery 4k",
            "тяжеловоз перевозка негабарита экскаватор вид сверху 4k"
        ],
        "shorts": [
            "overhead heavy haul oversize load #shorts",
            "top down truck carrying bulldozer #shorts",
            "oversize load semi truck drone #shorts",
            "road train oversize cargo #shorts"
        ]
    },
    "Самосвалы с породой/песком (Dump Trucks / Mining Hauling)": {
        "long": [
            "top down dump trucks hauling gravel dirt drone 4k",
            "overhead loaded tipper truck construction drone 4k",
            "mining dump truck hauling rocks aerial view 4k",
            "birds eye view kamaz dump truck hauling sand 4k",
            "самосвалы перевозка щебня песка вид сверху 4k"
        ],
        "shorts": [
            "top down dump truck hauling dirt #shorts",
            "overhead loaded dump truck 4k #shorts",
            "mining truck loaded aerial #shorts"
        ]
    },
    "Грузовики с грузом на бездорожье (Off-road / Mud / Extreme Loaded)": {
        "long": [
            "top down kamaz loaded mud extreme off road drone 4k",
            "overhead ural 4320 loaded deep mud 4k",
            "aerial view overloaded truck climbing steep dirt hill 4k",
            "zil 130 loaded off road mud drone 4k",
            "грузовики с грузом бездорожье грязь вид сверху 4k"
        ],
        "shorts": [
            "top down kamaz loaded mud #shorts",
            "overhead overloaded truck steep hill #shorts",
            "ural truck deep mud drone #shorts"
        ]
    },
    "Лесовозы с бревнами (Logging Trucks / Timber)": {
        "long": [
            "top down logging truck loaded with timber drone 4k",
            "overhead heavy loaded logging trucks highway drone",
            "birds eye view logging truck mud road 4k",
            "лесовоз с бревнами бездорожье вид сверху 4k"
        ],
        "shorts": [
            "top down logging truck loaded #shorts",
            "overhead heavy logging truck #shorts"
        ]
    },
    "Все виды (Фуры, Тяжеловозы, Самосвалы, Бездорожье)": {
        "long": TOP_DOWN_GLOBAL_QUERIES,
        "shorts": TOP_DOWN_SHORTS_GLOBAL_QUERIES
    }
}


def find_ffmpeg() -> Tuple[Optional[str], Optional[str]]:
    """
    Поиск исполняемых файлов ffmpeg и ffprobe в локальной папке bin или в системе.
    """
    project_bin = Path(__file__).resolve().parent / "bin"
    candidates_ffmpeg = [
        str(project_bin / "ffmpeg.exe"),
        "ffmpeg",
        shutil.which("ffmpeg") or "",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
    ]
    candidates_ffprobe = [
        str(project_bin / "ffprobe.exe"),
        "ffprobe",
        shutil.which("ffprobe") or "",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffprobe.exe"),
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\ProgramData\chocolatey\bin\ffprobe.exe"
    ]

    found_ffmpeg = None
    found_ffprobe = None

    for cmd in candidates_ffmpeg:
        if not cmd:
            continue
        try:
            res = subprocess.run([cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
            if res.returncode == 0:
                found_ffmpeg = cmd
                break
        except Exception:
            continue

    for cmd in candidates_ffprobe:
        if not cmd:
            continue
        try:
            res = subprocess.run([cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
            if res.returncode == 0:
                found_ffprobe = cmd
                break
        except Exception:
            continue

    return found_ffmpeg, found_ffprobe


class TruckVideoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Truck Video Cutter - 4K Drone & Top-Down Cargo Traffic Engine")
        self.geometry("1080x970")
        self.minsize(920, 780)

        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.yolo_model: Optional[YOLO] = None

        self.ffmpeg_path, self.ffprobe_path = find_ffmpeg()
        self.custom_cookie_path: Optional[str] = None
        self.learning_engine = AILearningEngine()

        # Инициализация интерфейса
        self._create_widgets()
        self._load_settings_preset()
        self._check_initial_system()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _create_widgets(self):
        # Главный контейнер
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Шапка (Заголовок приложения)
        header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e222d")
        header_frame.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="TRUCK & MILITARY CUTTER - GROUND & AIR ENGINE",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#38bdf8"
        )
        title_label.grid(row=0, column=0, padx=15, pady=(8, 2), sticky="w")

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Поиск по всему YouTube | Сортировка: 'грузовики', 'наземная' и 'воздушная' | Контроль водяных знаков | ИИ YOLOv8",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        subtitle_label.grid(row=1, column=0, padx=15, pady=(0, 8), sticky="w")

        # 2. Основная рабочая область (Панель настроек + Консоль)
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        main_content.grid_columnconfigure(0, weight=6)
        main_content.grid_columnconfigure(1, weight=4)
        main_content.grid_rowconfigure(0, weight=1)

        # Левая колонка: Настройки со скроллом
        left_frame = ctk.CTkScrollableFrame(main_content, corner_radius=10, label_text="Параметры нарезки и фильтрации")
        left_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        left_frame.grid_columnconfigure(1, weight=1)

        row_idx = 0

        # Панель пресетов настроек (Сохранить / Сбросить)
        preset_frame = ctk.CTkFrame(left_frame, fg_color="#181b22", corner_radius=8)
        preset_frame.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=(2, 8), sticky="ew")
        preset_frame.grid_columnconfigure(0, weight=3)
        preset_frame.grid_columnconfigure(1, weight=2)

        self.save_preset_btn = ctk.CTkButton(
            preset_frame,
            text="💾 Сохранить мой пресет",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0d9488",
            hover_color="#0f766e",
            height=30,
            command=lambda: self._save_settings_preset(silent=False)
        )
        self.save_preset_btn.grid(row=0, column=0, padx=(8, 4), pady=6, sticky="ew")

        self.reset_preset_btn = ctk.CTkButton(
            preset_frame,
            text="🔄 Сброс настроек",
            font=ctk.CTkFont(size=12),
            fg_color="#475569",
            hover_color="#334155",
            height=30,
            command=self._reset_settings_preset
        )
        self.reset_preset_btn.grid(row=0, column=1, padx=(4, 8), pady=6, sticky="ew")
        row_idx += 1

        # Источник видео
        ctk.CTkLabel(left_frame, text="Источник видео:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.source_mode_var = ctk.StringVar(value="Автопоиск YouTube")
        self.source_mode_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.source_mode_var,
            values=[
                "Автопоиск YouTube",
                "Своя ссылка YouTube",
                "Бесплатные стоки (Pexels / Pixabay)"
            ],
            command=self._on_source_mode_changed
        )
        self.source_mode_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Ракурс съемки (Вид сверху / Дрон)
        self.view_angle_label = ctk.CTkLabel(left_frame, text="Ракурс съемки:", font=ctk.CTkFont(weight="bold"))
        self.view_angle_label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.view_angle_var = ctk.StringVar(value="Только вид сверху / дрон (Top-Down / Aerial)")
        self.view_angle_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.view_angle_var,
            values=[
                "Только вид сверху / дрон (Top-Down / Aerial)",
                "Любой ракурс (Сверху, сбоку, с дороги)"
            ]
        )
        self.view_angle_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Тематика грузоперевозок / военной техники
        self.cargo_theme_label = ctk.CTkLabel(left_frame, text="Тематика видео / техники:", font=ctk.CTkFont(weight="bold"))
        self.cargo_theme_label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.cargo_theme_var = ctk.StringVar(value="Все виды техники (Автосортировка: грузовики / наземная / воздушная)")
        self.cargo_theme_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.cargo_theme_var,
            values=list(CARGO_THEMES.keys())
        )
        self.cargo_theme_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Формат источников YouTube (Длинные, Shorts или оба)
        self.yt_type_label = ctk.CTkLabel(left_frame, text="Формат видео YouTube:", font=ctk.CTkFont(weight="bold"))
        self.yt_type_label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.youtube_type_var = ctk.StringVar(value="Все форматы (Длинные видео и Shorts)")
        self.youtube_type_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.youtube_type_var,
            values=[
                "Все форматы (Длинные видео и Shorts)",
                "Только длинные видео (стандартные)",
                "Только YouTube Shorts (вертикальные ролики)"
            ],
            command=self._on_youtube_type_changed
        )
        self.youtube_type_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Поле ввода ссылки или поискового запроса
        self.query_label = ctk.CTkLabel(left_frame, text="Поисковый запрос / URL:", font=ctk.CTkFont(weight="bold"))
        self.query_label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
        self.query_entry = ctk.CTkEntry(
            left_frame,
            placeholder_text="Оставьте пустым для автопоиска по всему YouTube или введите свой запрос"
        )
        self.query_entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Блок API-ключей для стоков (по умолчанию скрыт)
        self.stock_frame = ctk.CTkFrame(left_frame, fg_color="#181b22", corner_radius=8)
        self.stock_frame.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.stock_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.stock_frame, text="API Pexels (опция):", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, padx=8, pady=4, sticky="w"
        )
        self.pexels_key_entry = ctk.CTkEntry(self.stock_frame, placeholder_text="Вставьте бесплатный ключ Pexels", show="*")
        self.pexels_key_entry.grid(row=0, column=1, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(self.stock_frame, text="API Pixabay (опция):", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, padx=8, pady=4, sticky="w"
        )
        self.pixabay_key_entry = ctk.CTkEntry(self.stock_frame, placeholder_text="Вставьте бесплатный ключ Pixabay", show="*")
        self.pixabay_key_entry.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        row_idx += 1
        self.stock_frame.grid_remove()

        # Качество загрузки видео с YouTube
        ctk.CTkLabel(left_frame, text="Качество загрузки:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.quality_var = ctk.StringVar(value="1080p Full HD (Стабильно, быстро)")
        self.quality_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.quality_var,
            values=[
                "1080p Full HD (Стабильно, быстро)",
                "720p HD (Экономный трафик)",
                "1440p / 4K (Максимум, дольше скачивается)"
            ]
        )
        self.quality_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Блок авторизации и обхода анти-бот защиты YouTube (Cookies / OAuth2)
        self.auth_frame = ctk.CTkFrame(left_frame, fg_color="#181b22", corner_radius=8)
        self.auth_frame.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.auth_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.auth_frame,
            text="Обход блокировок YouTube (Cookies):",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38bdf8"
        ).grid(row=0, column=0, columnspan=2, padx=8, pady=(6, 2), sticky="w")

        auth_sub = ctk.CTkFrame(self.auth_frame, fg_color="transparent")
        auth_sub.grid(row=1, column=0, columnspan=2, padx=8, pady=3, sticky="ew")
        auth_sub.grid_columnconfigure(0, weight=1)

        self.auth_mode_var = ctk.StringVar(value="Авто (файл cookies.txt при наличии)")
        self.auth_mode_menu = ctk.CTkOptionMenu(
            auth_sub,
            variable=self.auth_mode_var,
            values=[
                "Авто (файл cookies.txt при наличии)",
                "Файл cookies.txt (выбрать файл...)",
                "Браузер: Firefox",
                "Браузер: Chrome",
                "Браузер: Edge",
                "Отключено (анонимно)"
            ],
            command=self._on_auth_mode_changed
        )
        self.auth_mode_menu.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.pick_cookies_btn = ctk.CTkButton(
            auth_sub,
            text="Выбрать файл...",
            width=110,
            command=self._on_pick_cookies
        )
        self.pick_cookies_btn.grid(row=0, column=1, padx=(0, 4))

        self.help_cookies_btn = ctk.CTkButton(
            auth_sub,
            text="?",
            width=32,
            fg_color="#334155",
            hover_color="#475569",
            command=self._show_cookie_instructions
        )
        self.help_cookies_btn.grid(row=0, column=2, padx=0)

        self.cookie_status_label = ctk.CTkLabel(
            self.auth_frame,
            text="Проверка cookies...",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        self.cookie_status_label.grid(row=2, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="w")
        row_idx += 1

        # Фильтр по дате публикации видео
        ctk.CTkLabel(left_frame, text="Дата публикации:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.date_filter_var = ctk.StringVar(value="За всё время (без ограничений)")
        self.date_filter_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.date_filter_var,
            values=[
                "За всё время (без ограничений)",
                "За последний месяц (30 дней)",
                "За последние 3 месяца",
                "За последние 6 месяцев",
                "За последний год (12 месяцев)",
                "За последние 2 года",
                "За последние 3 года"
            ]
        )
        self.date_filter_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Фильтр длительности исходника (мин)
        ctk.CTkLabel(left_frame, text="Длительность источника:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        dur_filter_subframe = ctk.CTkFrame(left_frame, fg_color="transparent")
        dur_filter_subframe.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(dur_filter_subframe, text="От:", font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.min_duration_entry = ctk.CTkEntry(dur_filter_subframe, width=55)
        self.min_duration_entry.insert(0, "0.1")  # ~6 сек для поддержки Shorts
        self.min_duration_entry.grid(row=0, column=1, padx=(0, 8), sticky="w")

        ctk.CTkLabel(dur_filter_subframe, text="мин.   До:", font=ctk.CTkFont(size=12)).grid(row=0, column=2, padx=(0, 4), sticky="w")
        self.max_duration_entry = ctk.CTkEntry(dur_filter_subframe, width=65)
        self.max_duration_entry.insert(0, "60")
        self.max_duration_entry.grid(row=0, column=3, padx=(0, 4), sticky="w")

        ctk.CTkLabel(dur_filter_subframe, text="мин (0=без лимита)", font=ctk.CTkFont(size=11), text_color="#94a3b8").grid(
            row=0, column=4, padx=(4, 0), sticky="w"
        )
        row_idx += 1

        # Формат выходного видео
        ctk.CTkLabel(left_frame, text="Формат видео:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.format_mode_var = ctk.StringVar(value="Вертикальное 9:16 (1080x1920 - Shorts/Reels/TikTok)")
        self.format_mode_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.format_mode_var,
            values=[
                "Вертикальное 9:16 (1080x1920 - Shorts/Reels/TikTok)",
                "Горизонтальное 16:9 (1920x1080 - YouTube)",
                "Оба формата (сохранять 16:9 и 9:16 в разные папки)"
            ]
        )
        self.format_mode_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Количество клипов
        ctk.CTkLabel(left_frame, text="Количество клипов:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.clip_count_entry = ctk.CTkEntry(left_frame, width=120)
        self.clip_count_entry.insert(0, "5")
        self.clip_count_entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
        row_idx += 1

        # Длительность клипа (для длинных видео)
        ctk.CTkLabel(left_frame, text="Длительность клипа (сек):", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.clip_duration_entry = ctk.CTkEntry(left_frame, width=120)
        self.clip_duration_entry.insert(0, "60")
        self.clip_duration_entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
        row_idx += 1

        # Папка сохранения
        ctk.CTkLabel(left_frame, text="Папка сохранения:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        folder_subframe = ctk.CTkFrame(left_frame, fg_color="transparent")
        folder_subframe.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        folder_subframe.grid_columnconfigure(0, weight=1)

        default_out = str((Path(__file__).resolve().parent / "output_videos"))
        self.output_dir_entry = ctk.CTkEntry(folder_subframe)
        self.output_dir_entry.insert(0, default_out)
        self.output_dir_entry.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="ew")

        self.browse_btn = ctk.CTkButton(
            folder_subframe,
            text="Выбрать...",
            width=80,
            command=self._on_browse_folder
        )
        self.browse_btn.grid(row=0, column=1, padx=0, pady=0)
        row_idx += 1

        # Фильтр водяных знаков и логотипов
        ctk.CTkLabel(left_frame, text="Фильтр водяных знаков:", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        self.watermark_mode_var = ctk.StringVar(value="Строгий фильтр (Только 100% чистые видео без знаков)")
        self.watermark_mode_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.watermark_mode_var,
            values=[
                "Строгий фильтр (Только 100% чистые видео без знаков)",
                "Умный режим (Автообрезка краев + отсев титров)",
                "Отключить фильтр водяных знаков"
            ],
            command=self._on_watermark_mode_changed
        )
        self.watermark_mode_menu.grid(row=row_idx, column=1, padx=10, pady=5, sticky="ew")
        row_idx += 1

        # Базовый зум кадра
        ctk.CTkLabel(left_frame, text="Базовый зум кадра (%):", font=ctk.CTkFont(weight="bold")).grid(
            row=row_idx, column=0, padx=10, pady=(6, 2), sticky="w"
        )
        self.crop_val_label = ctk.CTkLabel(left_frame, text="0% (Полный оригинальный кадр)")
        self.crop_val_label.grid(row=row_idx, column=1, padx=10, pady=(6, 2), sticky="w")
        row_idx += 1

        self.crop_slider = ctk.CTkSlider(
            left_frame,
            from_=0,
            to=25,
            number_of_steps=25,
            command=self._on_crop_slider_change
        )
        self.crop_slider.set(0)
        self.crop_slider.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="ew")
        row_idx += 1

        # Чекбокс: ИИ-фильтр грузовиков и грузов
        self.ai_filter_var = ctk.BooleanVar(value=True)
        self.ai_filter_check = ctk.CTkCheckBox(
            left_frame,
            text="ИИ-фильтр грузовиков и грузов (YOLOv8)",
            variable=self.ai_filter_var,
            font=ctk.CTkFont(weight="bold")
        )
        self.ai_filter_check.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=4, sticky="w")
        row_idx += 1

        # Режим детекции ИИ
        ctk.CTkLabel(left_frame, text="Режим ИИ-детекции:", font=ctk.CTkFont(size=12)).grid(
            row=row_idx, column=0, padx=10, pady=4, sticky="w"
        )
        self.ai_mode_var = ctk.StringVar(value="Грузовики, военная техника и дроны (Автосортировка: грузовики/наземная/воздушная)")
        self.ai_mode_menu = ctk.CTkOptionMenu(
            left_frame,
            variable=self.ai_mode_var,
            values=[
                "Грузовики, военная техника и дроны (Автосортировка: грузовики/наземная/воздушная)",
                "Грузовики и военная техника + Дроны (Автосортировка наземная/воздушная)",
                "Военная наземная техника (Танки, БТР, Артиллерия, Грузовики)",
                "Дроны и БПЛА (Воздушная техника, FPV, Квадрокоптеры)",
                "Грузовики с грузом / кузовом / прицепом (Рекомендуется)",
                "Любые грузовики и спецтехника (Без проверки груза)",
                "Грузовики + общий дорожный трафик"
            ]
        )
        self.ai_mode_menu.grid(row=row_idx, column=1, padx=10, pady=4, sticky="ew")
        row_idx += 1

        # Блок Самообучения ИИ
        learn_frame = ctk.CTkFrame(left_frame, fg_color="#181b22", corner_radius=8)
        learn_frame.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        learn_frame.grid_columnconfigure(0, weight=1)

        self.ai_learning_var = ctk.BooleanVar(value=True)
        self.ai_learning_check = ctk.CTkCheckBox(
            learn_frame,
            text="🧠 Самообучение ИИ на скачиваемых видео",
            variable=self.ai_learning_var,
            font=ctk.CTkFont(weight="bold")
        )
        self.ai_learning_check.grid(row=0, column=0, padx=8, pady=(6, 2), sticky="w")

        learn_sub = ctk.CTkFrame(learn_frame, fg_color="transparent")
        learn_sub.grid(row=1, column=0, padx=8, pady=(0, 6), sticky="ew")
        learn_sub.grid_columnconfigure(0, weight=1)

        self.ai_learning_status_label = ctk.CTkLabel(
            learn_sub,
            text="Нейропамять: загрузка...",
            font=ctk.CTkFont(size=11),
            text_color="#38bdf8"
        )
        self.ai_learning_status_label.grid(row=0, column=0, sticky="w")

        self.ai_learning_btn = ctk.CTkButton(
            learn_sub,
            text="📊 База знаний ИИ",
            width=130,
            height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._show_learning_stats
        )
        self.ai_learning_btn.grid(row=0, column=1, sticky="e")
        row_idx += 1

        # Защита от повторов: История скачиваний
        history_cnt = len(load_download_history())
        hist_subframe = ctk.CTkFrame(left_frame, fg_color="transparent")
        hist_subframe.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        hist_subframe.grid_columnconfigure(0, weight=1)

        self.avoid_repeats_var = ctk.BooleanVar(value=True)
        self.history_check = ctk.CTkCheckBox(
            hist_subframe,
            text=f"Исключать повторы (В истории: {history_cnt} видео)",
            variable=self.avoid_repeats_var,
            font=ctk.CTkFont(weight="bold")
        )
        self.history_check.grid(row=0, column=0, sticky="w")

        self.clear_hist_btn = ctk.CTkButton(
            hist_subframe,
            text="Сбросить историю",
            width=120,
            fg_color="#475569",
            hover_color="#334155",
            command=self._on_clear_history
        )
        self.clear_hist_btn.grid(row=0, column=1, padx=(6, 0), sticky="e")
        row_idx += 1

        # Удаление звука
        self.remove_audio_var = ctk.BooleanVar(value=True)
        self.remove_audio_check = ctk.CTkCheckBox(
            left_frame,
            text="Удалить звук из видео (чистый видеоряд без АП на музыку)",
            variable=self.remove_audio_var
        )
        self.remove_audio_check.grid(row=row_idx, column=0, columnspan=2, padx=10, pady=6, sticky="w")
        row_idx += 1

        # Правая колонка: Живой лог и статистика
        right_frame = ctk.CTkFrame(main_content, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkLabel(
            right_frame,
            text="Журнал работы (Консоль)",
            font=ctk.CTkFont(weight="bold")
        )
        log_header.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            right_frame,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#0f172a",
            text_color="#e2e8f0"
        )
        self.log_textbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # 3. Нижняя панель управления: Прогресс-бар и Кнопки действий
        bottom_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1e222d")
        bottom_frame.grid(row=2, column=0, padx=15, pady=(10, 15), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Статус и процент
        status_line = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_line.grid(row=0, column=0, padx=15, pady=(10, 4), sticky="ew")
        status_line.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_line,
            text="Готов к работе",
            font=ctk.CTkFont(weight="bold"),
            text_color="#94a3b8"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.percent_label = ctk.CTkLabel(
            status_line,
            text="0%",
            font=ctk.CTkFont(weight="bold"),
            text_color="#38bdf8"
        )
        self.percent_label.grid(row=0, column=1, sticky="e")

        # Прогресс-бар
        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=15, pady=4, sticky="ew")

        # Кнопки
        btn_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=15, pady=(8, 12), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Запустить скачивание и нарезку",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=42,
            command=self._on_start_clicked
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹ Остановить",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=42,
            state="disabled",
            command=self._on_stop_clicked
        )
        self.stop_btn.grid(row=0, column=1, padx=6, sticky="ew")

        self.open_folder_btn = ctk.CTkButton(
            btn_frame,
            text="📁 Открыть папку с клипами",
            font=ctk.CTkFont(size=13),
            fg_color="#334155",
            hover_color="#475569",
            height=42,
            command=self._on_open_output_folder
        )
        self.open_folder_btn.grid(row=0, column=2, padx=(6, 0), sticky="ew")

    def _check_initial_system(self):
        self.log("Приложение инициализировано.")
        if self.ffmpeg_path:
            self.log(f"FFmpeg успешно обнаружен: {self.ffmpeg_path}")
        else:
            self.log("[ПРЕДУПРЕЖДЕНИЕ] FFmpeg не обнаружен в PATH или папке bin!")
            self.log("Убедитесь, что FFmpeg установлен в системе.")
        self._update_cookie_status()
        self._update_learning_ui_status()

    def _update_learning_ui_status(self):
        """Обновление статуса нейропамяти в интерфейсе."""
        try:
            summary = self.learning_engine.get_stats_summary()
            q_cnt = summary["total_queries_tracked"]
            ch_cnt = summary["total_channels_rated"]
            sq_cnt = summary["smart_queries_count"]
            txt = f"Опыт: {summary['total_clips']} клипов | Запросов: {q_cnt} | Каналов: {ch_cnt} | Умных: {sq_cnt}"
            self.ai_learning_status_label.configure(text=txt)
        except Exception:
            pass

    def _show_learning_stats(self):
        """Отображение подробной статистики и базы знаний самообучения ИИ."""
        summary = self.learning_engine.get_stats_summary()

        top_q_str = "\n".join([f"  • {q} (Балл: {score}, Клипов: {clips})" for q, score, clips in summary["top_queries"]]) or "  (Пока нет накопленных данных)"
        top_c_str = "\n".join([f"  • {ch} (Рейтинг: {rating}, Клипов: {clips})" for ch, rating, clips in summary["top_channels"]]) or "  (Пока нет данных по каналам)"
        top_kw_str = ", ".join([f"{kw} ({weight})" for kw, weight in summary["top_keywords"][:6]]) or "(Нет)"
        smart_q_str = "\n".join([f"  • {sq}" for sq in summary["smart_queries"]]) or "  (Сформируются автоматически после первых скачиваний)"

        msg = (
            f"🧠 СТАТИСТИКА САМООБУЧЕНИЯ ИИ:\n\n"
            f"Всего проанализировано видео: {summary['total_analyzed']}\n"
            f"Успешно нарезано чистых клипов: {summary['total_clips']}\n"
            f"Отслеживаемых запросов в памяти: {summary['total_queries_tracked']}\n"
            f"Оцененных каналов YouTube: {summary['total_channels_rated']}\n\n"
            f"ТОП-5 РЕЗУЛЬТАТИВНЫХ ЗАПРОСОВ:\n{top_q_str}\n\n"
            f"ТОП ПРОВЕРЕННЫХ КАНАЛОВ С ВЫСОКИМ КАЧЕСТВОМ:\n{top_c_str}\n\n"
            f"ЛУЧШИЕ КЛЮЧЕВЫЕ ПАТТЕРНЫ (N-граммы):\n{top_kw_str}\n\n"
            f"СИНТЕЗИРОВАННЫЕ УМНЫЕ ЗАПРОСЫ (SMART QUERIES):\n{smart_q_str}\n\n"
            f"Сбросить накопленную память самообучения?"
        )

        if messagebox.askyesno("База знаний самообучения ИИ", msg):
            self.learning_engine.reset_memory()
            self._update_learning_ui_status()
            self.log("[ИИ Обучение] Память самообучения успешно сброшена до начального состояния.")

    def _update_cookie_status(self):
        active = find_active_cookie_file(self.custom_cookie_path)
        if active:
            f_size = os.path.getsize(active) / 1024
            name = Path(active).name
            self.cookie_status_label.configure(
                text=f"✓ Cookies активны: {name} ({f_size:.1f} KB)",
                text_color="#10b981"
            )
            self.log(f"[YouTube Cookies] Готов к работе файл cookies (формат Netscape): {name} ({f_size:.1f} KB).")
        else:
            self.cookie_status_label.configure(
                text="cookies.txt не найден (для защиты от ошибки бота нажмите '?')",
                text_color="#f59e0b"
            )

    def _on_auth_mode_changed(self, choice: str):
        if "выбрать файл" in choice.lower():
            self._on_pick_cookies()

    def _on_pick_cookies(self):
        selected = filedialog.askopenfilename(
            title="Выберите файл cookies для YouTube",
            filetypes=[("Файлы cookies", "*.txt"), ("Все файлы", "*.*")]
        )
        if selected and os.path.isfile(selected):
            valid = ensure_netscape_cookies(selected)
            if valid:
                self.custom_cookie_path = valid
                self.auth_mode_var.set("Файл cookies.txt (выбрать файл...)")
                self._update_cookie_status()
            else:
                messagebox.showwarning(
                    "Неверный файл cookies",
                    "Выбранный файл не содержит распознаваемых cookies YouTube.\n"
                    "Экспортируйте cookies с сайта youtube.com через расширение 'Get cookies.txt locally'."
                )

    def _show_cookie_instructions(self):
        msg = (
            "КАК УБРАТЬ ОШИБКУ 'Sign in to confirm you’re not a bot':\n\n"
            "1. Установите расширение 'Get cookies.txt locally' в Chrome, Edge или Firefox.\n"
            "2. Зайдите на https://www.youtube.com (убедитесь, что вы залогинены).\n"
            "3. Нажмите на значок расширения и экспортируйте cookies в файл 'cookies.txt'.\n"
            "4. Положите файл 'cookies.txt' в папку программы или нажмите кнопку 'Выбрать файл...'.\n\n"
            "После этого скачивание работает со 100% стабильностью и максимальной скоростью без блокировок!\n"
            "Подробная инструкция также открыта в файле КАК_ОБОЙТИ_БЛОКИРОВКУ_YOUTUBE.txt."
        )
        messagebox.showinfo("Инструкция по cookies YouTube", msg)

    def log(self, message: str):
        """Потокобезопасный вывод логов в консоль приложения."""
        def append():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_textbox.insert("end", f"[{timestamp}] {message}\n")
            self.log_textbox.see("end")
        self.after(0, append)

    def set_status(self, text: str, progress: Optional[float] = None):
        """Обновление статуса и прогресс-бара."""
        def update():
            self.status_label.configure(text=text)
            if progress is not None:
                p_clamped = max(0.0, min(1.0, progress))
                self.progress_bar.set(p_clamped)
                self.percent_label.configure(text=f"{int(p_clamped * 100)}%")
        self.after(0, update)

    def _update_history_ui_count(self):
        cnt = len(load_download_history())
        self.history_check.configure(text=f"Исключать повторы (В истории: {cnt} видео)")

    def _on_clear_history(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить историю скачанных видео?"):
            clear_download_history_file()
            self._update_history_ui_count()
            self.log("[История] База истории успешно очищена.")

    def _get_current_settings(self) -> Dict[str, Any]:
        """Сбор всех текущих настроек интерфейса в словарь."""
        return {
            "source_mode": self.source_mode_var.get(),
            "view_angle": self.view_angle_var.get(),
            "cargo_theme": self.cargo_theme_var.get(),
            "youtube_type": self.youtube_type_var.get(),
            "query": self.query_entry.get().strip(),
            "pexels_key": self.pexels_key_entry.get().strip(),
            "pixabay_key": self.pixabay_key_entry.get().strip(),
            "quality": self.quality_var.get(),
            "auth_mode": self.auth_mode_var.get(),
            "custom_cookie_path": self.custom_cookie_path,
            "date_filter": self.date_filter_var.get(),
            "min_duration": self.min_duration_entry.get().strip(),
            "max_duration": self.max_duration_entry.get().strip(),
            "format_mode": self.format_mode_var.get(),
            "clip_count": self.clip_count_entry.get().strip(),
            "clip_duration": self.clip_duration_entry.get().strip(),
            "output_dir": self.output_dir_entry.get().strip(),
            "watermark_mode": self.watermark_mode_var.get(),
            "crop_slider": int(self.crop_slider.get()),
            "ai_filter": bool(self.ai_filter_var.get()),
            "ai_mode": self.ai_mode_var.get(),
            "ai_learning": bool(self.ai_learning_var.get()),
            "avoid_repeats": bool(self.avoid_repeats_var.get()),
            "remove_audio": bool(self.remove_audio_var.get()),
        }

    def _apply_settings(self, settings: Dict[str, Any]):
        """Применение словаря настроек к элементам интерфейса."""
        try:
            if "source_mode" in settings:
                self.source_mode_var.set(settings["source_mode"])
                self._on_source_mode_changed(settings["source_mode"])

            if "view_angle" in settings:
                self.view_angle_var.set(settings["view_angle"])

            if "cargo_theme" in settings:
                self.cargo_theme_var.set(settings["cargo_theme"])

            if "youtube_type" in settings:
                self.youtube_type_var.set(settings["youtube_type"])

            if "query" in settings:
                self.query_entry.delete(0, "end")
                self.query_entry.insert(0, str(settings["query"]))

            if "pexels_key" in settings:
                self.pexels_key_entry.delete(0, "end")
                self.pexels_key_entry.insert(0, str(settings["pexels_key"]))

            if "pixabay_key" in settings:
                self.pixabay_key_entry.delete(0, "end")
                self.pixabay_key_entry.insert(0, str(settings["pixabay_key"]))

            if "quality" in settings:
                self.quality_var.set(settings["quality"])

            if "custom_cookie_path" in settings and settings["custom_cookie_path"]:
                cand = str(settings["custom_cookie_path"])
                if os.path.isfile(cand):
                    self.custom_cookie_path = cand

            if "auth_mode" in settings:
                self.auth_mode_var.set(settings["auth_mode"])

            if "date_filter" in settings:
                self.date_filter_var.set(settings["date_filter"])

            if "min_duration" in settings:
                self.min_duration_entry.delete(0, "end")
                self.min_duration_entry.insert(0, str(settings["min_duration"]))

            if "max_duration" in settings:
                self.max_duration_entry.delete(0, "end")
                self.max_duration_entry.insert(0, str(settings["max_duration"]))

            if "format_mode" in settings:
                self.format_mode_var.set(settings["format_mode"])

            if "clip_count" in settings:
                self.clip_count_entry.delete(0, "end")
                self.clip_count_entry.insert(0, str(settings["clip_count"]))

            if "clip_duration" in settings:
                self.clip_duration_entry.delete(0, "end")
                self.clip_duration_entry.insert(0, str(settings["clip_duration"]))

            if "output_dir" in settings:
                self.output_dir_entry.delete(0, "end")
                self.output_dir_entry.insert(0, str(settings["output_dir"]))

            if "watermark_mode" in settings:
                self.watermark_mode_var.set(settings["watermark_mode"])

            if "crop_slider" in settings:
                val = float(settings["crop_slider"])
                self.crop_slider.set(val)
                self._on_crop_slider_change(val)

            if "ai_filter" in settings:
                self.ai_filter_var.set(bool(settings["ai_filter"]))

            if "ai_mode" in settings:
                self.ai_mode_var.set(settings["ai_mode"])

            if "ai_learning" in settings:
                self.ai_learning_var.set(bool(settings["ai_learning"]))

            if "avoid_repeats" in settings:
                self.avoid_repeats_var.set(bool(settings["avoid_repeats"]))

            if "remove_audio" in settings:
                self.remove_audio_var.set(bool(settings["remove_audio"]))
        except Exception as e:
            self.log(f"[Пресет] Ошибка при применении настроек: {e}")

    def _save_settings_preset(self, silent: bool = False):
        """Сохранение текущих настроек в JSON файл."""
        try:
            data = self._get_current_settings()
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if not silent:
                self.log(f"[Пресет] Настройки успешно сохранены в {SETTINGS_FILE.name}")
                messagebox.showinfo("Пресет сохранен", f"Текущий пресет настроек успешно сохранен!\n\nФайл: {SETTINGS_FILE.name}")
        except Exception as e:
            if not silent:
                self.log(f"[Пресет] Ошибка сохранения настроек: {e}")
                messagebox.showerror("Ошибка", f"Не удалось сохранить пресет: {e}")

    def _load_settings_preset(self):
        """Загрузка сохраненного пресета настроек при старте приложения."""
        if not SETTINGS_FILE.is_file():
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._apply_settings(data)
                self.log(f"[Пресет] Загружен сохраненный пресет настроек ({SETTINGS_FILE.name}).")
        except Exception as e:
            self.log(f"[Пресет] Не удалось загрузить сохраненный пресет: {e}")

    def _reset_settings_preset(self):
        """Сброс настроек до заводских параметров по умолчанию."""
        if messagebox.askyesno("Сброс настроек", "Сбросить все параметры до настроек по умолчанию?"):
            self._apply_settings(DEFAULT_SETTINGS)
            self._save_settings_preset(silent=True)
            self.log("[Пресет] Все настройки сброшены до значений по умолчанию.")

    def _on_window_close(self):
        """Автосохранение настроек при закрытии окна приложения."""
        try:
            self._save_settings_preset(silent=True)
        except Exception:
            pass
        self.destroy()

    def _on_source_mode_changed(self, choice: str):
        if "стоки" in choice.lower():
            self.stock_frame.grid()
            self.auth_frame.grid_remove()
            self.view_angle_label.grid_remove()
            self.view_angle_menu.grid_remove()
            self.cargo_theme_label.grid_remove()
            self.cargo_theme_menu.grid_remove()
            self.yt_type_label.grid_remove()
            self.youtube_type_menu.grid_remove()
            self.query_label.configure(text="Поисковый запрос на стоках:")
        elif "своя ссылка" in choice.lower():
            self.stock_frame.grid_remove()
            self.auth_frame.grid()
            self.view_angle_label.grid()
            self.view_angle_menu.grid()
            self.cargo_theme_label.grid_remove()
            self.cargo_theme_menu.grid_remove()
            self.yt_type_label.grid_remove()
            self.youtube_type_menu.grid_remove()
            self.query_label.configure(text="Прямая ссылка YouTube:")
        else:
            self.stock_frame.grid_remove()
            self.auth_frame.grid()
            self.view_angle_label.grid()
            self.view_angle_menu.grid()
            self.cargo_theme_label.grid()
            self.cargo_theme_menu.grid()
            self.yt_type_label.grid()
            self.youtube_type_menu.grid()
            self.query_label.configure(text="Поисковый запрос YouTube:")

    def _on_youtube_type_changed(self, choice: str):
        """Автоматическая подстройка фильтра длительности под выбранный тип видео."""
        choice_lower = choice.lower()
        if "только youtube shorts" in choice_lower:
            self.min_duration_entry.delete(0, "end")
            self.min_duration_entry.insert(0, "0.1")  # 6 сек
            self.max_duration_entry.delete(0, "end")
            self.max_duration_entry.insert(0, "1")    # 60 сек
        elif "только длинные" in choice_lower:
            self.min_duration_entry.delete(0, "end")
            self.min_duration_entry.insert(0, "1")    # 60 сек
            self.max_duration_entry.delete(0, "end")
            self.max_duration_entry.insert(0, "60")
        else:  # Все форматы
            self.min_duration_entry.delete(0, "end")
            self.min_duration_entry.insert(0, "0.1")  # от 6 сек
            self.max_duration_entry.delete(0, "end")
            self.max_duration_entry.insert(0, "60")

    def _on_watermark_mode_changed(self, choice: str):
        choice_lower = choice.lower()
        if "строгий" in choice_lower:
            self.crop_slider.set(0)
            self.crop_val_label.configure(text="0% (Полный оригинальный кадр)")
        elif "умный" in choice_lower:
            self.crop_slider.set(10)
            self.crop_val_label.configure(text="10% (Анти-вотермарк зум)")

    def _on_crop_slider_change(self, val: float):
        v = int(val)
        if v == 0:
            self.crop_val_label.configure(text="0% (Полный оригинальный кадр)")
        else:
            self.crop_val_label.configure(text=f"{v}% (Зум кадра)")

    def _on_browse_folder(self):
        selected = filedialog.askdirectory(title="Выберите папку для сохранения готовых клипов")
        if selected:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, selected)

    def _on_open_output_folder(self):
        out_dir = self.output_dir_entry.get().strip()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # Гарантируем наличие подпапок для категорий
        os.makedirs(os.path.join(out_dir, "грузовики"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "наземная"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "воздушная"), exist_ok=True)
        if sys.platform == "win32":
            os.startfile(out_dir)
        else:
            subprocess.run(["xdg-open", out_dir])

    def _on_start_clicked(self):
        # Автоматическое сохранение текущего пресета перед запуском
        self._save_settings_preset(silent=True)

        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Внимание", "Процесс уже выполняется!")
            return

        # Валидация входных данных
        try:
            target_clips = int(self.clip_count_entry.get().strip())
            clip_duration = int(self.clip_duration_entry.get().strip())
            if target_clips <= 0 or clip_duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Укажите корректные числа для количества клипов и длительности клипа!")
            return

        # Парсинг длительности исходника
        try:
            min_src_min = float(self.min_duration_entry.get().strip() or "0")
            max_src_min = float(self.max_duration_entry.get().strip() or "0")
            if min_src_min < 0 or max_src_min < 0:
                raise ValueError
            if max_src_min > 0 and min_src_min > max_src_min:
                messagebox.showerror("Ошибка", "Минимальная длительность источника не может быть больше максимальной!")
                return
        except ValueError:
            messagebox.showerror("Ошибка", "Укажите корректные числа для фильтра длительности источника (мин)!")
            return

        min_duration_sec = max(4, int(min_src_min * 60))
        max_duration_sec = int(max_src_min * 60) if max_src_min > 0 else 0

        # Парсинг фильтра по дате
        cutoff_date = None
        date_choice = self.date_filter_var.get()
        now = datetime.now()
        if "1 месяц" in date_choice or ("последний месяц" in date_choice and "3" not in date_choice and "6" not in date_choice):
            cutoff_date = now - timedelta(days=30)
        elif "3 месяца" in date_choice:
            cutoff_date = now - timedelta(days=90)
        elif "6 месяцев" in date_choice:
            cutoff_date = now - timedelta(days=180)
        elif "1 год" in date_choice or "последний год" in date_choice:
            cutoff_date = now - timedelta(days=365)
        elif "2 года" in date_choice:
            cutoff_date = now - timedelta(days=730)
        elif "3 года" in date_choice:
            cutoff_date = now - timedelta(days=1095)

        out_dir = self.output_dir_entry.get().strip()
        if not out_dir:
            messagebox.showerror("Ошибка", "Укажите папку для сохранения клипов!")
            return

        # Перепроверяем FFmpeg
        self.ffmpeg_path, self.ffprobe_path = find_ffmpeg()
        if not self.ffmpeg_path:
            messagebox.showerror(
                "Ошибка FFmpeg",
                "FFmpeg не найден в системе!\n\nПожалуйста, установите FFmpeg или поместите ffmpeg.exe в папку 'bin'."
            )
            return

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.set_status("Запуск процесса...", 0.0)

        # Сбор параметров
        config = {
            "source_mode": self.source_mode_var.get(),
            "view_angle": self.view_angle_var.get(),
            "cargo_theme": self.cargo_theme_var.get(),
            "youtube_type": self.youtube_type_var.get(),
            "query_text": self.query_entry.get().strip(),
            "youtube_quality": self.quality_var.get(),
            "date_filter_name": self.date_filter_var.get(),
            "cutoff_date": cutoff_date,
            "min_duration_sec": min_duration_sec,
            "max_duration_sec": max_duration_sec,
            "format_mode": self.format_mode_var.get(),
            "target_clips": target_clips,
            "clip_duration": clip_duration,
            "output_dir": out_dir,
            "crop_percent": int(self.crop_slider.get()),
            "watermark_mode": self.watermark_mode_var.get(),
            "check_watermarks": "отключ" not in self.watermark_mode_var.get().lower(),
            "use_ai_filter": self.ai_filter_var.get(),
            "ai_mode": self.ai_mode_var.get(),
            "use_ai_learning": self.ai_learning_var.get(),
            "avoid_repeats": self.avoid_repeats_var.get(),
            "remove_audio": self.remove_audio_var.get(),
            "cookie_file": find_active_cookie_file(self.custom_cookie_path),
            "auth_mode": self.auth_mode_var.get(),
            "pexels_key": self.pexels_key_entry.get().strip(),
            "pixabay_key": self.pixabay_key_entry.get().strip(),
            "ffmpeg": self.ffmpeg_path,
            "ffprobe": self.ffprobe_path
        }

        self.worker_thread = threading.Thread(target=self._run_processing_pipeline, args=(config,), daemon=True)
        self.worker_thread.start()

    def _on_stop_clicked(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.log("[ОСТАНОВКА] Запрос остановки процесса. Завершаем текущую операцию...")
            self.set_status("Остановка процесса...")
            self.stop_btn.configure(state="disabled")

    def _run_processing_pipeline(self, config: Dict[str, Any]):
        try:
            pipeline = ProcessingPipeline(self, config, self.stop_event)
            pipeline.run()
        except Exception as e:
            self.log(f"[ОШИБКА] Произошел сбой: {str(e)}")
        finally:
            self.after(0, self._on_pipeline_finished)

    def _on_pipeline_finished(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._update_history_ui_count()
        self._update_learning_ui_status()
        if self.stop_event.is_set():
            self.set_status("Процесс был остановлен пользователем.")
            self.log("Процесс остановлен.")
        else:
            self.set_status("Готово!", 1.0)
            self.log("Обработка успешно завершена!")


class ProcessingPipeline:
    def __init__(self, app: TruckVideoApp, config: Dict[str, Any], stop_event: threading.Event):
        self.app = app
        self.cfg = config
        self.stop_event = stop_event
        self.temp_dir = Path(__file__).resolve().parent / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(self.cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Гарантируем создание папок для грузовиков, наземной и воздушной техники
        (self.out_dir / "грузовики").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "наземная").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "воздушная").mkdir(parents=True, exist_ok=True)

        # Загрузка базы истории и модуля самообучения ИИ
        self.history = load_download_history()
        self.learning_engine: Optional[AILearningEngine] = getattr(self.app, 'learning_engine', None)
        self._cookie_logged = False
        self.active_cookie_file = self.cfg.get("cookie_file") or find_active_cookie_file()

        if "оба формата" in self.cfg["format_mode"].lower():
            (self.out_dir / "грузовики" / "9_16").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "грузовики" / "16_9").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "наземная" / "9_16").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "наземная" / "16_9").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "воздушная" / "9_16").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "воздушная" / "16_9").mkdir(parents=True, exist_ok=True)

    def _build_yt_dlp_opts(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Формирование безопасных параметров yt-dlp для обхода анти-бот защиты и решения JS-челленджей."""
        opts: Dict[str, Any] = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'file_access_retries': 5,
            'http_chunk_size': 5242880,
            'remote_components': ['ejs:github'],
            'js_runtimes': {'node': {'path': None}},
        }

        # 1. Проверяем файл cookies (самый надёжный способ)
        active_cookie = find_active_cookie_file(self.cfg.get("cookie_file"))
        if active_cookie and os.path.isfile(active_cookie) and os.path.getsize(active_cookie) > 10:
            opts['cookiefile'] = active_cookie
            if not self._cookie_logged:
                self.app.log(f"[YouTube] Активна авторизация через cookies (Netscape): {Path(active_cookie).name}")
                self._cookie_logged = True
                self.active_cookie_file = active_cookie
        else:
            # 2. Проверяем выбранный режим браузера / авторизации
            auth_mode = self.cfg.get("auth_mode", "Авто").lower()
            if "firefox" in auth_mode and "авто" not in auth_mode:
                opts['cookiesfrombrowser'] = ('firefox',)
            elif "chrome" in auth_mode:
                opts['cookiesfrombrowser'] = ('chrome',)
            elif "edge" in auth_mode:
                opts['cookiesfrombrowser'] = ('edge',)
            elif "opera" in auth_mode:
                opts['cookiesfrombrowser'] = ('opera',)

        if extra:
            opts.update(extra)
        return opts

    def check_blacklist(self, title: str, description: str = "", tags: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Строгая проверка на игры, симуляторы, анимации и превью со стоков с водяными знаками."""
        full_text = f"{title} {description} {' '.join(tags or [])}".lower()
        
        # 1. Проверка на симуляторы/игры
        for bad_word in STRICT_BLACKLIST:
            pattern = r'\b' + re.escape(bad_word) + r'\b'
            if re.search(pattern, full_text):
                return False, f"Обнаружено нежелательное слово '{bad_word}' (игра/симулятор)"

        # 2. Проверка на стоковые превью с водяными знаками (если включен фильтр водяных знаков)
        wm_mode = self.cfg.get("watermark_mode", "").lower()
        if "отключ" not in wm_mode:
            for wm_meta in WATERMARK_META_BLACKLIST:
                if wm_meta in full_text:
                    return False, f"Обнаружен стоковый водяной знак в метаданных ('{wm_meta}')"

            # При "Строгом фильтре" дополнительно отсекаем видео со словами 'stock footage' / 'stock video'
            if "строгий" in wm_mode:
                if re.search(r'\bstock footage\b|\bstock video\b|\bfootage preview\b', full_text):
                    return False, "Обнаружен маркер стокового видео ('stock footage/video' в строгом режиме)"

        return True, "OK"

    def _classify_video_category(self, title: str = "", description: str = "", query: str = "",
                                yolo_counts: Optional[Dict[str, int]] = None) -> str:
        """
        Классификация контента на три категории:
        - 'грузовики': грузовики, фуры, тягачи, самосвалы, лесовозы, тяжеловозы, коммерческий транспорт
        - 'наземная': военная наземная техника (танки, БМП, БТР, военные грузовики, колонны, РСЗО, артиллерия)
        - 'воздушная': дроны, БПЛА, квадрокоптеры, беспилотники, FPV, барражирующие боеприпасы
        """
        full_text = f"{title} {description} {query}".lower()

        air_score = 0
        truck_score = 0
        military_score = 0

        # 1. Сильные воздушные маркеры (FPV, БПЛА, квадрокоптер, ланцет, bayraktar, shahed...)
        for kw in AIR_STRONG_KEYWORDS:
            if kw in full_text:
                air_score += 4

        # 2. Мягкие воздушные маркеры (дрон, drone, аэросъемка)
        for kw in AIR_SOFT_KEYWORDS:
            if kw in full_text:
                air_score += 1

        # 3. Маркеры грузовиков (фуры, тягачи, самосвалы, лесовозы, тяжеловозы, прицепы...)
        for kw in TRUCK_KEYWORDS:
            if kw in full_text:
                truck_score += 4

        # 4. Маркеры военной наземной техники (танки, БМП, БТР, военные колонны, РСЗО...)
        for kw in MILITARY_GROUND_STRONG_KEYWORDS:
            if kw in full_text:
                military_score += 4

        # 5. Бренды техники (Камаз, Урал, Краз, Scania, Volvo, MAN...)
        for brand in TRUCK_AND_GROUND_BRANDS:
            if brand in full_text:
                if military_score > 0 or "военн" in full_text or "армейск" in full_text:
                    military_score += 2
                else:
                    truck_score += 2

        # 6. Учет выбранной пользователем тематики
        theme = self.cfg.get("cargo_theme", "").lower()
        if "дроны" in theme and "наземная" not in theme and "грузовик" not in theme:
            air_score += 6
        elif "военная наземная" in theme or ("наземная" in theme and "дроны" not in theme and "фуры" not in theme and "грузовик" not in theme):
            military_score += 6
        elif any(t in theme for t in ["фуры", "тяжеловозы", "самосвалы", "бездорожье", "лесовозы", "грузовики"]):
            truck_score += 6

        # 7. Учет результатов ИИ-детекции YOLOv8
        if yolo_counts:
            truck_objs = yolo_counts.get("truck", 0)
            ground_objs = yolo_counts.get("ground", 0)
            air_objs = yolo_counts.get("air", 0)

            truck_score += truck_objs * 3
            military_score += max(0, ground_objs - truck_objs) * 2
            air_score += air_objs * 3

        # Определение категории:
        # Если явные маркеры дронов/БПЛА превышают грузовики и военную технику -> воздушная
        if air_score > truck_score and air_score > military_score:
            return "воздушная"
        # Если грузовики преобладают над военной техникой -> грузовики
        elif truck_score > military_score:
            return "грузовики"
        # Если военная техника преобладает над грузовиками -> наземная
        elif military_score > truck_score:
            return "наземная"
        else:
            # При равенстве очков:
            if "военн" in theme or "военн" in full_text:
                return "наземная"
            return "грузовики"

    def run(self):
        self.app.log("==========================================")
        self.app.log(f"Начало обработки. Цель: {self.cfg['target_clips']} клипов.")
        self.app.log(f"Источник: {self.cfg['source_mode']}")
        if "автопоиск" in self.cfg["source_mode"].lower():
            self.app.log(f"Ракурс: {self.cfg['view_angle']}")
            self.app.log(f"Тематика грузов: {self.cfg['cargo_theme']}")
            self.app.log(f"Формат YouTube: {self.cfg['youtube_type']}")
            self.app.log(f"Исключать повторы: {'Да (история активна)' if self.cfg['avoid_repeats'] else 'Нет'}")
        wm_info = self.cfg.get('watermark_mode', 'Строгий фильтр')
        self.app.log(f"Фильтр водяных знаков: {wm_info}")
        self.app.log(f"Качество YouTube: {self.cfg['youtube_quality']}")
        self.app.log(f"Фильтр по дате: {self.cfg['date_filter_name']}")
        max_dur_txt = f"{int(self.cfg['max_duration_sec'] // 60)} мин" if self.cfg['max_duration_sec'] > 0 else "без ограничений"
        min_dur_txt = f"{self.cfg['min_duration_sec']} сек" if self.cfg['min_duration_sec'] < 60 else f"{int(self.cfg['min_duration_sec'] // 60)} мин"
        self.app.log(f"Длительность источника: от {min_dur_txt} до {max_dur_txt}")

        # Очистка старых временных файлов перед стартом
        self._cleanup_temp()

        # Инициализация модели YOLOv8
        yolo_model = None
        if self.cfg["use_ai_filter"]:
            self.app.set_status("Загрузка ИИ-модели YOLOv8...", 0.05)
            self.app.log("Инициализация нейросети YOLOv8n (imgsz=960 для детекции грузовиков и кузовов)...")
            local_pt = Path(__file__).resolve().parent / "yolov8n.pt"
            try:
                yolo_model = YOLO(str(local_pt))
                self.app.log("YOLOv8 успешно загружена и готова к детекции.")
            except Exception as e:
                self.app.log(f"Ошибка загрузки YOLOv8: {e}. Продолжение без ИИ-фильтра...")
                yolo_model = None

        saved_clips_count = 0
        target = self.cfg["target_clips"]
        source_mode = self.cfg["source_mode"]

        if "автопоиск youtube" in source_mode.lower():
            saved_clips_count = self._process_youtube_search(yolo_model, target)
        elif "своя ссылка youtube" in source_mode.lower():
            saved_clips_count = self._process_direct_youtube(yolo_model, target)
        elif "стоки" in source_mode.lower():
            saved_clips_count = self._process_stock_videos(yolo_model, target)

        self.app.log("==========================================")
        self.app.log(f"Итог: Сохранено клипов: {saved_clips_count} из {target}")
        self._cleanup_temp()

    def _process_youtube_search(self, yolo_model: Optional[YOLO], target: int) -> int:
        saved = 0
        yt_type = self.cfg.get("youtube_type", "").lower()
        theme_name = self.cfg.get("cargo_theme", "Все виды (Фуры, Тяжеловозы, Самосвалы, Бездорожье)")
        theme_data = CARGO_THEMES.get(theme_name, CARGO_THEMES["Все виды (Фуры, Тяжеловозы, Самосвалы, Бездорожье)"])
        top_down_only = "сверху" in self.cfg.get("view_angle", "").lower()

        # Построение глубокого пула поисковых запросов
        if self.cfg["query_text"]:
            queries = [self.cfg["query_text"]]
        elif "только youtube shorts" in yt_type:
            queries = theme_data["shorts"] + TOP_DOWN_SHORTS_GLOBAL_QUERIES
        elif "только длинные" in yt_type:
            queries = theme_data["long"] + TOP_DOWN_GLOBAL_QUERIES
        else:
            # Смешанный режим: чередование длинных запросов и Shorts
            queries = []
            pool_long = theme_data["long"] + TOP_DOWN_GLOBAL_QUERIES
            pool_shorts = theme_data["shorts"] + TOP_DOWN_SHORTS_GLOBAL_QUERIES
            max_len = max(len(pool_long), len(pool_shorts))
            for i in range(max_len):
                if i < len(pool_long) and pool_long[i] not in queries:
                    queries.append(pool_long[i])
                if i < len(pool_shorts) and pool_shorts[i] not in queries:
                    queries.append(pool_shorts[i])

        # Оптимизация пула запросов с помощью самообучения ИИ
        if self.cfg.get("use_ai_learning", True) and self.learning_engine and not self.cfg["query_text"]:
            original_len = len(queries)
            queries = self.learning_engine.rank_and_augment_queries(queries, theme_name=theme_name)
            self.app.log(f"[ИИ Обучение] Пул запросов адаптирован нейропамятью: {len(queries)} вариантов (было {original_len}), топ-запросы выдвинуты вперёд.")

        # Цикл по пулу запросов до набора необходимого числа клипов
        consecutive_bot_fails = 0

        for query in queries:
            if saved >= target or self.stop_event.is_set():
                break

            self.app.log(f"Поиск по YouTube (вид сверху): '{query}'...")
            self.app.set_status(f"Поиск YouTube: {query}...")

            ydl_opts = self._build_yt_dlp_opts({'extract_flat': 'in_playlist'})

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Глубокий поиск до 40 видео на каждый запрос
                    search_res = ydl.extract_info(f"ytsearch40:{query}", download=False)
                    entries = search_res.get('entries', []) if search_res else []
            except Exception as e:
                self.app.log(f"Ошибка поиска yt-dlp: {e}")
                continue

            # Сортировка видеокандидатов с помощью обученного классификатора
            if self.cfg.get("use_ai_learning", True) and self.learning_engine and len(entries) > 1:
                entries = sorted(
                    entries,
                    key=lambda e: self.learning_engine.score_candidate_entry(
                        e.get('title', ''),
                        e.get('uploader') or e.get('channel', ''),
                        e.get('description', '')
                    ),
                    reverse=True
                )

            self.app.log(f"Найдено видеокандидатов: {len(entries)}")

            for entry in entries:
                if saved >= target or self.stop_event.is_set():
                    break
                if not entry:
                    continue

                vid_id = entry.get('id', '')
                vid_title = entry.get('title', 'Без названия')
                vid_url = entry.get('url') or entry.get('webpage_url') or f"https://www.youtube.com/watch?v={vid_id}"

                # 1. Защита от повторов: Проверка истории
                if self.cfg["avoid_repeats"] and vid_id in self.history:
                    self.app.log(f"[История] Ролик уже скачивался ранее: '{vid_title}' ({vid_id}). Пропуск.")
                    continue

                # 2. Фильтр стоп-слов
                is_safe, reason = self.check_blacklist(vid_title, entry.get('description', ''))
                if not is_safe:
                    self.app.log(f"[Фильтр] Пропущено ({reason}): '{vid_title}'")
                    if self.cfg.get("use_ai_learning", True) and self.learning_engine:
                        self.learning_engine.record_failure(
                            query=query,
                            video_title=vid_title,
                            channel=entry.get('uploader') or entry.get('channel', ''),
                            reason=reason
                        )
                    continue

                # 3. Фильтр по длительности
                dur = entry.get('duration')
                min_dur = self.cfg["min_duration_sec"]
                max_dur = self.cfg["max_duration_sec"]

                if "только youtube shorts" in yt_type and dur and dur > 70:
                    continue
                if "только длинные" in yt_type and dur and dur < 60:
                    continue

                if dur is not None and dur > 0:
                    if dur < min_dur:
                        self.app.log(f"[Фильтр длит.] Пропущено ({int(dur)} сек < {min_dur} сек): '{vid_title}'")
                        continue
                    if max_dur > 0 and dur > max_dur:
                        self.app.log(f"[Фильтр длит.] Пропущено ({int(dur // 60)} мин > {int(max_dur // 60)} мин): '{vid_title}'")
                        continue

                # 4. Фильтр по дате публикации (если включен)
                cutoff_date = self.cfg["cutoff_date"]
                if cutoff_date:
                    self.app.set_status(f"Проверка даты: {vid_title[:35]}...")
                    try:
                        check_opts = self._build_yt_dlp_opts({'extract_flat': False})
                        with yt_dlp.YoutubeDL(check_opts) as info_ydl:
                            f_info = info_ydl.extract_info(vid_url, download=False)
                            up_date_str = f_info.get('upload_date')
                            if up_date_str:
                                vdate = datetime.strptime(up_date_str, '%Y%m%d')
                                if vdate < cutoff_date:
                                    self.app.log(f"[Фильтр даты] Пропущено (загружено {vdate.strftime('%d.%m.%Y')}): '{vid_title}'")
                                    continue
                            full_dur = f_info.get('duration')
                            if full_dur:
                                if full_dur < min_dur or (max_dur > 0 and full_dur > max_dur):
                                    continue
                    except Exception as e:
                        err_msg = str(e)
                        if "sign in to confirm" in err_msg.lower() or "bot" in err_msg.lower():
                            consecutive_bot_fails += 1
                            if consecutive_bot_fails <= 2:
                                self.app.log(f"[YouTube Bot Check] Блокировка анонимного запроса для '{vid_title}'.")
                            if consecutive_bot_fails >= 3 and not self.active_cookie_file:
                                self.app.log("------------------------------------------------------------")
                                self.app.log("[ВНИМАНИЕ] YouTube блокирует запросы: 'Sign in to confirm you’re not a bot'.")
                                self.app.log("Для продолжения добавьте файл cookies.txt (кнопка '?' в окне программы).")
                                self.app.log("------------------------------------------------------------")
                                break
                            continue
                        self.app.log(f"Не удалось получить дату для '{vid_title}': {e}")

                self.app.log(f"Выбран кандидат: '{vid_title}'")
                source_file = self._download_youtube_video(vid_url)
                if not source_file or not os.path.exists(source_file):
                    consecutive_bot_fails += 1
                    if consecutive_bot_fails >= 3 and not self.active_cookie_file:
                        self.app.log("------------------------------------------------------------")
                        self.app.log("[ОСТАНОВКА] Несколько видео подряд заблокированы защитой YouTube.")
                        self.app.log("YouTube отклоняет скачивание с вашего IP без файла cookies.txt.")
                        self.app.log("Экспортируйте cookies.txt из браузера (кнопка '?' в окне программы)")
                        self.app.log("и перезапустите скачивание.")
                        self.app.log("------------------------------------------------------------")
                        break
                    self.app.log(f"Не удалось загрузить '{vid_title}'. Переход к следующему...")
                    continue

                consecutive_bot_fails = 0

                # 5. Проверка ракурса (вид сверху) и обработка
                created = self._slice_and_filter_video(
                    source_file,
                    yolo_model,
                    target - saved,
                    top_down_only,
                    video_title=vid_title,
                    video_desc=entry.get('description', ''),
                    query=query
                )
                saved += created

                # Обучение на результате обработки видео
                if self.cfg.get("use_ai_learning", True) and self.learning_engine:
                    if created > 0:
                        new_smart = self.learning_engine.record_success(
                            query=query,
                            video_title=vid_title,
                            channel=entry.get('uploader') or entry.get('channel', ''),
                            tags=entry.get('tags'),
                            clips_count=created
                        )
                        self.app.log(f"[ИИ Обучение] Успешный опыт закреплен для '{query}' (+{created} клипов).")
                        if new_smart:
                            self.app.log(f"[ИИ Обучение] Сгенерирован новый умный запрос: '{new_smart[0]}'")
                    else:
                        self.learning_engine.record_failure(
                            query=query,
                            video_title=vid_title,
                            channel=entry.get('uploader') or entry.get('channel', ''),
                            reason="Клипы отклонены фильтром водяных знаков или ИИ"
                        )

                # Сохраняем в историю, если видео было обработано
                if vid_id and self.cfg["avoid_repeats"]:
                    save_download_history_entry(vid_id, vid_title)
                    self.history[vid_id] = {"title": vid_title}

                try:
                    if os.path.exists(source_file):
                        os.remove(source_file)
                except Exception:
                    pass

        return saved

    def _process_direct_youtube(self, yolo_model: Optional[YOLO], target: int) -> int:
        url = self.cfg["query_text"]
        if not url:
            self.app.log("Ошибка: Ссылка на YouTube не указана!")
            return 0

        self.app.log(f"Анализ прямой ссылки: {url}")
        self.app.set_status("Анализ видео...")

        top_down_only = "сверху" in self.cfg.get("view_angle", "").lower()
        vid_id = ""

        ydl_opts = self._build_yt_dlp_opts({'extract_flat': False})
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                vid_id = info.get('id', '')
                vid_title = info.get('title', 'Видео')

                # Проверка истории
                if self.cfg["avoid_repeats"] and vid_id in self.history:
                    self.app.log(f"[История] Это видео уже есть в истории скачанных: '{vid_title}'")

                is_safe, reason = self.check_blacklist(vid_title, info.get('description', ''))
                if not is_safe:
                    self.app.log(f"[Фильтр] Видео отклонено ({reason}): '{vid_title}'")
                    return 0

                dur = info.get('duration')
                min_dur = self.cfg["min_duration_sec"]
                max_dur = self.cfg["max_duration_sec"]
                if dur is not None and dur > 0:
                    if dur < min_dur or (max_dur > 0 and dur > max_dur):
                        self.app.log(f"[Фильтр длит.] Видео отклонено по длительности ({int(dur)} сек).")
                        return 0

                cutoff_date = self.cfg["cutoff_date"]
                if cutoff_date and info.get('upload_date'):
                    vdate = datetime.strptime(info['upload_date'], '%Y%m%d')
                    if vdate < cutoff_date:
                        self.app.log(f"[Фильтр даты] Видео старше выбранного фильтра. Отклонено.")
                        return 0

        except Exception as e:
            self.app.log(f"Не удалось получить информацию о видео: {e}")
            return 0

        source_file = self._download_youtube_video(url)
        if not source_file or not os.path.exists(source_file):
            return 0

        saved = self._slice_and_filter_video(
            source_file,
            yolo_model,
            target,
            top_down_only,
            video_title=info.get('title', 'Видео'),
            video_desc=info.get('description', ''),
            query=url
        )
        if vid_id and self.cfg["avoid_repeats"]:
            save_download_history_entry(vid_id, info.get('title', 'Видео'))
            self.history[vid_id] = {"title": info.get('title', 'Видео')}

        # Обучение на результате обработки прямой ссылки
        if self.cfg.get("use_ai_learning", True) and self.learning_engine:
            vid_title = info.get('title', 'Видео')
            if saved > 0:
                new_smart = self.learning_engine.record_success(
                    query=url,
                    video_title=vid_title,
                    channel=info.get('uploader') or info.get('channel', ''),
                    tags=info.get('tags'),
                    clips_count=saved
                )
                self.app.log(f"[ИИ Обучение] Успешный опыт закреплен для прямой ссылки (+{saved} клипов).")
                if new_smart:
                    self.app.log(f"[ИИ Обучение] Сгенерирован новый умный запрос: '{new_smart[0]}'")
            else:
                self.learning_engine.record_failure(
                    query=url,
                    video_title=vid_title,
                    channel=info.get('uploader') or info.get('channel', ''),
                    reason="Клипы отклонены фильтром водяных знаков или ИИ"
                )

        try:
            if os.path.exists(source_file):
                os.remove(source_file)
        except Exception:
            pass
        return saved

    def _process_stock_videos(self, yolo_model: Optional[YOLO], target: int) -> int:
        saved = 0
        pexels_key = self.cfg.get("pexels_key", "")
        pixabay_key = self.cfg.get("pixabay_key", "")
        query = self.cfg["query_text"] or "top down semi trucks highway drone"
        top_down_only = "сверху" in self.cfg.get("view_angle", "").lower()

        if not pexels_key and not pixabay_key:
            self.app.log("[Внимание] Не указаны API-ключи для Pexels или Pixabay! Переключаемся на автопоиск YouTube...")
            return self._process_youtube_search(yolo_model, target)

        if pexels_key and saved < target and not self.stop_event.is_set():
            self.app.log(f"Запрос к Pexels API: '{query}'...")
            try:
                headers = {"Authorization": pexels_key}
                r = requests.get(
                    f"https://api.pexels.com/videos/search?query={requests.utils.quote(query)}&per_page=15&orientation=landscape",
                    headers=headers,
                    timeout=15
                )
                if r.status_code == 200:
                    data = r.json()
                    videos = data.get("videos", [])
                    self.app.log(f"Pexels вернул {len(videos)} видео.")
                    for v in videos:
                        if saved >= target or self.stop_event.is_set():
                            break
                        v_dur = v.get("duration", 0)
                        if v_dur and v_dur < 4:
                            continue

                        files = v.get("video_files", [])
                        files.sort(key=lambda x: x.get("width", 0), reverse=True)
                        if not files or not files[0].get("link"):
                            continue

                        temp_dest = str(self.temp_dir / f"pexels_{v.get('id', int(time.time()))}.mp4")
                        self.app.log(f"Скачивание стокового видео Pexels #{v.get('id')}...")
                        if self._download_direct_file(files[0]["link"], temp_dest):
                            created = self._slice_and_filter_video(
                                temp_dest,
                                yolo_model,
                                target - saved,
                                top_down_only,
                                video_title=f"pexels_{v.get('id', '')}",
                                video_desc=query,
                                query=query
                            )
                            saved += created
                            try:
                                if os.path.exists(temp_dest):
                                    os.remove(temp_dest)
                            except Exception:
                                pass
            except Exception as e:
                self.app.log(f"Ошибка запроса Pexels: {e}")

        return saved

    def _download_direct_file(self, url: str, dest_path: str) -> bool:
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if self.stop_event.is_set():
                            return False
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            self.app.log(f"Ошибка скачивания: {e}")
            return False

    def _cleanup_partial(self, timestamp: int):
        try:
            for p in self.temp_dir.glob(f"src_{timestamp}*"):
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _download_youtube_video(self, url: str) -> Optional[str]:
        timestamp = int(time.time())
        output_template = str(self.temp_dir / f"src_{timestamp}.%(ext)s")

        self.app.log("Подключение к YouTube...")
        self.app.set_status("Скачивание YouTube видео...")

        quality_pref = self.cfg.get("youtube_quality", "1080p")
        if "720" in quality_pref:
            fmt = "bestvideo[height<=720]+bestaudio/best[height<=720]/bestvideo*+bestaudio/best"
        elif "1440" in quality_pref or "4k" in quality_pref.lower():
            fmt = "bestvideo+bestaudio/bestvideo*+bestaudio/best"
        else:
            fmt = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo*+bestaudio/best"

        def progress_hook(d):
            if self.stop_event.is_set():
                raise yt_dlp.utils.DownloadCancelled("Загрузка остановлена пользователем")
            if d.get('status') == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded = d.get('downloaded_bytes', 0)
                speed = d.get('speed', 0) or 0
                speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 1024*1024 else f"{speed / 1024:.0f} KB/s"
                if total > 0:
                    pct = int(downloaded / total * 100)
                    self.app.set_status(f"Загрузка: {pct}% ({speed_str})")
                else:
                    mb = downloaded / (1024*1024)
                    self.app.set_status(f"Загрузка: {mb:.1f} MB ({speed_str})")
            elif d.get('status') == 'finished':
                self.app.set_status("Сборка и проверка видеофайла...")

        extra = {
            'format': fmt,
            'outtmpl': output_template,
            'progress_hooks': [progress_hook],
            'ffmpeg_location': self.cfg["ffmpeg"],
            'merge_output_format': 'mp4'
        }
        ydl_opts = self._build_yt_dlp_opts(extra)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            expected_path = Path(self.temp_dir / f"src_{timestamp}.mp4")
            if expected_path.exists() and expected_path.stat().st_size > 500 * 1024:
                size_mb = expected_path.stat().st_size / (1024 * 1024)
                self.app.log(f"Видео успешно скачано ({size_mb:.1f} MB).")
                return str(expected_path)

            for p in self.temp_dir.glob(f"src_{timestamp}.*"):
                if p.is_file() and p.suffix not in [".part", ".ytdl"] and p.stat().st_size > 500 * 1024:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    self.app.log(f"Видео успешно скачано ({size_mb:.1f} MB).")
                    return str(p)

            self.app.log("[Ошибка] Скачанный файл пустой или поврежден.")
            self._cleanup_partial(timestamp)
            return None
        except Exception as e:
            err_str = str(e)
            if "остановлена пользователем" in err_str.lower():
                self.app.log("Загрузка отменена.")
            elif "sign in to confirm" in err_str.lower() or "bot" in err_str.lower():
                self.app.log("------------------------------------------------------------")
                self.app.log("[ОШИБКА БЛОКИРОВКИ YOUTUBE: BOT-CHECK]")
                self.app.log("YouTube требует подтверждение: 'Sign in to confirm you’re not a bot'.")
                self.app.log("КАК РЕШИТЬ:")
                self.app.log("1. Экспортируйте cookies YouTube расширением 'Get cookies.txt locally'")
                self.app.log("2. Сохраните 'cookies.txt' в папку программы (или нажмите 'Выбрать файл...')")
                self.app.log("3. Подробная инструкция в файле 'КАК_ОБОЙТИ_БЛОКИРОВКУ_YOUTUBE.txt'")
                self.app.log("------------------------------------------------------------")
            else:
                self.app.log(f"Ошибка при скачивании YouTube: {e}")
            self._cleanup_partial(timestamp)
            return None

    def _get_ffprobe_duration(self, video_path: str) -> float:
        if self.cfg.get("ffprobe") and os.path.exists(video_path):
            try:
                cmd = [
                    self.cfg["ffprobe"], "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                val = float(res.stdout.strip())
                if val > 0:
                    return val
            except Exception:
                pass
        return 0.0

    def _get_video_duration(self, video_path: str) -> float:
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return 0.0
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._get_ffprobe_duration(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
            return self._get_ffprobe_duration(video_path)
        except Exception:
            return self._get_ffprobe_duration(video_path)

    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass
        return 1920, 1080

    def _is_top_down_view(self, frame: np.ndarray) -> bool:
        """
        Проверка ракурса съемки: вид сверху (дрон/аэросъемка) или съемка с земли/горизонт.
        В виде сверху кадр заполнен дорожным полотном и окружающим рельефом без открытого неба.
        """
        try:
            h, w = frame.shape[:2]
            top_region = frame[0:int(h * 0.30), :]
            gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_top, 50, 150)
            edge_density = float(np.mean(edges > 0))

            hsv_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            is_bright_sky = float(np.mean(hsv_top[:, :, 2])) > 195 and float(np.mean(hsv_top[:, :, 1])) < 60

            # Если в верхней части кадра плотное небо без деталей дороги - это ракурс с земли
            if is_bright_sky and edge_density < 0.008:
                return False
            return True
        except Exception:
            return True

    def _check_text_clustering(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[bool, int]:
        """
        Проверка кучности: поиск >= 3 горизонтально выстроенных символов/слов с близкой высотой и базовой линией.
        """
        if len(boxes) < 3:
            return False, 0
        boxes = sorted(boxes, key=lambda b: b[0])
        for i in range(len(boxes)):
            line_cluster = [boxes[i]]
            base_cy = boxes[i][1]
            base_ch = boxes[i][3]
            for j in range(i + 1, len(boxes)):
                cx, cy, cw, ch = boxes[j]
                h_diff = abs(ch - base_ch) / max(ch, base_ch, 1)
                y_diff = abs(cy - base_cy) / max(base_ch, 1)
                if h_diff <= 0.30 and y_diff <= 0.35:
                    prev_cx, prev_cy, prev_cw, prev_ch = line_cluster[-1]
                    gap = cx - (prev_cx + prev_cw)
                    if -5 <= gap <= max(45, base_ch * 2.5):
                        line_cluster.append(boxes[j])
                        if len(line_cluster) >= 3:
                            return True, len(line_cluster)
        return False, 0

    def _detect_watermarks_and_subtitles(self, frame: np.ndarray) -> Tuple[bool, bool, str, float, float]:
        """
        Точный градиентно-контрастный анализ кадра на водяные знаки, логотипы, титры и субтитры.
        Возвращает: (has_watermark: bool, is_center: bool, reason: str, depth: float, confidence: float)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Новые калиброванные пороги ROI (0.25 для углов, 0.26 для субтитров/заголовков, 0.32 для центра)
        top_h = int(h * 0.22)
        bot_h = int(h * 0.24)
        corner_w = int(w * 0.28)
        sub_x1 = int(w * 0.08)
        sub_x2 = int(w * 0.92)

        rois = [
            ("верхний левый угол", 0, top_h, 0, corner_w, False, 0.25),
            ("верхний правый угол", 0, top_h, w - corner_w, w, False, 0.25),
            ("нижний левый угол", h - bot_h, h, 0, corner_w, False, 0.25),
            ("нижний правый угол", h - bot_h, h, w - corner_w, w, False, 0.25),
            ("субтитры (низ)", h - bot_h, int(h * 0.98), sub_x1, sub_x2, False, 0.26),
            ("заголовок (верх)", int(h * 0.02), top_h, sub_x1, sub_x2, False, 0.26),
            ("центр кадра", int(h * 0.25), int(h * 0.75), int(w * 0.15), int(w * 0.85), True, 0.32),
        ]

        detections = []
        max_confidence = 0.0
        has_center = False
        max_depth = 0.0

        for name, y1, y2, x1, x2, is_center, threshold in rois:
            roi_gray = gray[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]
            rh, rw = roi_gray.shape[:2]
            if rh < 15 or rw < 15:
                continue

            # 2. Морфологический градиент (выделение границ шрифтов и мелкой графики)
            grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grad = cv2.morphologyEx(roi_gray, cv2.MORPH_GRADIENT, grad_kernel)
            _, grad_thresh = cv2.threshold(grad, 38, 255, cv2.THRESH_BINARY)

            canny_roi = cv2.Canny(roi_gray, 30, 95)

            # 3. Цветовая изоляция (белый/серый, желтый, красный и голубой/синий)
            v_chan = roi_hsv[:, :, 2]
            s_chan = roi_hsv[:, :, 1]
            h_chan = roi_hsv[:, :, 0]

            white_mask = (v_chan >= 145) & (s_chan <= 80)
            yellow_mask = (h_chan >= 14) & (h_chan <= 40) & (s_chan >= 65) & (v_chan >= 120)
            cyan_red_mask = ((h_chan <= 12) | (h_chan >= 165) | ((h_chan >= 75) & (h_chan <= 135))) & (s_chan >= 85) & (v_chan >= 110)
            color_mask = (white_mask | yellow_mask | cyan_red_mask).astype(np.uint8) * 255
            dilated_color = cv2.dilate(color_mask, grad_kernel)

            # Пересечение градиента/граней с цветовой маской для отсечения дорожного шума
            feature_map = cv2.bitwise_and(cv2.bitwise_or(grad_thresh, canny_roi), dilated_color)

            # 4. Умеренное ядро склейки букв (9, 3) и RETR_LIST для обнаружения букв внутри плашек/рамок
            conn_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            connected = cv2.morphologyEx(feature_map, cv2.MORPH_CLOSE, conn_kernel)

            contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            roi_best_score = 0.0
            roi_best_desc = ""
            candidate_boxes = []

            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                area = cw * ch

                # Защита от склейки и потери крупных титров: допустимая высота до int(rh * 0.75), площадь до (rw * rh * 0.65)
                if area < 35 or area > (rw * rh * 0.65):
                    continue
                if cw < 10 or ch < 6 or ch > int(rh * 0.75):
                    continue

                aspect = cw / max(1, ch)
                c_canny = canny_roi[cy:cy+ch, cx:cx+cw]
                edge_dens = float(np.mean(c_canny > 0))

                c_bright = color_mask[cy:cy+ch, cx:cx+cw]
                bright_dens = float(np.mean(c_bright > 0))

                roi_crop = roi_gray[cy:cy+ch, cx:cx+cw]
                contrast = float(np.std(roi_crop))

                # Диапазоны для текста (вытянутый) и компактного логотипа/иконки
                is_text_shape = (1.15 <= aspect <= 30.0) and (edge_dens >= 0.05 or bright_dens >= 0.08) and (contrast >= 8.5)
                is_logo_shape = (0.50 <= aspect <= 1.55) and (cw >= 12 and ch >= 8) and (edge_dens >= 0.06 or bright_dens >= 0.10) and (contrast >= 10.0)

                if is_text_shape or is_logo_shape:
                    candidate_boxes.append((cx, cy, cw, ch))
                    score = min(0.99, 0.30 + (edge_dens * 1.8) + (bright_dens * 0.5) + (contrast / 130.0))
                    if score > roi_best_score:
                        roi_best_score = score
                        kind = "текст/субтитры" if is_text_shape else "логотип/водяной знак"
                        roi_best_desc = f"{kind} ({cw}x{ch}, score={score:.2f})"

            # 5. Проверка кучности (Text Line / Cluster Check)
            has_cluster, cluster_cnt = self._check_text_clustering(candidate_boxes)
            if has_cluster:
                cluster_score = 0.90
                if cluster_score > roi_best_score:
                    roi_best_score = cluster_score
                    roi_best_desc = f"текстовая строка из {cluster_cnt} символов (score={cluster_score:.2f})"

            if roi_best_score >= threshold:
                max_confidence = max(max_confidence, roi_best_score)
                detections.append(f"{name}: {roi_best_desc}")
                if is_center:
                    has_center = True
                    max_depth = max(max_depth, 0.5)
                else:
                    if "верх" in name:
                        max_depth = max(max_depth, top_h / h)
                    elif "нижн" in name or "субтитры" in name:
                        max_depth = max(max_depth, bot_h / h)

        has_wm = len(detections) > 0
        reason_str = "; ".join(detections) if detections else "чистый кадр"
        return has_wm, has_center, reason_str, max_depth, max_confidence

    def _verify_clip_watermarks_and_angle(self, video_path: str, top_down_only: bool) -> Tuple[bool, int]:
        """
        Комплексная проверка ролика на водяные знаки, титры и ракурс съемки:
        1. Межкадровый детектор стабильности граней (Temporal Persistence): находит статичные оверлеи/логотипы.
        2. Покадровый градиентно-цветовой анализ: находит субтитры, текстовые плашки и водяные знаки.
        3. Если включен строгий режим: ролик отклоняется при наличии ЛЮБОГО знака/логотипа/титров.
        4. Если включен умный режим: краевые знаки обрезаются зумом, центральные титры отсеиваются.
        При ЛЮБОМ сбое cv2.VideoCapture или checked_frames_count == 0 гарантированно возвращает False, 0.
        """
        video_name = Path(video_path).name
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.app.log(f"[Watermark Debug] ✗ Ошибка: не удалось открыть видеопоток '{video_name}'. Файл поврежден или недоступен. Вердикт: REJECTED.")
                return False, 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                self.app.log(f"[Watermark Debug] ✗ Ошибка: нулевое количество кадров в видео '{video_name}'. Вердикт: REJECTED.")
                return False, 0

            # Равномерная выборка 7 кадров по всей длине ролика (10%, 25%, 40%, 50%, 65%, 80%, 92%)
            sample_percentages = [10, 25, 40, 50, 65, 80, 92]
            sample_indices = [int(total_frames * (pct / 100.0)) for pct in sample_percentages]

            center_text_hits = 0
            watermark_hits = 0
            detected_reasons = []
            top_down_hits = 0
            max_depth_seen = 0.0
            peak_confidence = 0.0
            checked_frames_count = 0
            frames_list = []

            self.app.log(f"[Watermark Debug] ─── Анализ водяных знаков: '{video_name}' (всего кадров: {total_frames}) ───")

            # 1. Извлечение кадров и покадровая детекция
            for idx, pct in zip(sample_indices, sample_percentages):
                if idx < 0 or idx >= total_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.app.log(f"[Watermark Debug] Кадр {idx} ({pct}%): Ошибка чтения кадра (None).")
                    continue

                checked_frames_count += 1
                frames_list.append((idx, pct, frame))

                # Проверка ракурса сверху
                if top_down_only:
                    if self._is_top_down_view(frame):
                        top_down_hits += 1

                # Покадровый детектор водяных знаков и титров
                if self.cfg.get("check_watermarks", True):
                    has_wm, is_center, reason, depth, conf = self._detect_watermarks_and_subtitles(frame)
                    peak_confidence = max(peak_confidence, conf)
                    status_str = "НАЙДЕН ЗНАК [FRAME OCR/SHAPE]" if has_wm else "ЧИСТО (PASS)"
                    self.app.log(f"[Watermark Debug] Кадр {idx} ({pct}%): Score={conf:.2f} | Вердикт: {status_str} | Детали: {reason}")

                    if has_wm:
                        watermark_hits += 1
                        detected_reasons.append(reason)
                        max_depth_seen = max(max_depth_seen, depth)
                        if is_center:
                            center_text_hits += 1

            if checked_frames_count == 0:
                self.app.log(f"[Watermark Debug] ИТОГ: REJECTED ✗ ('{video_name}': не удалось прочитать ни одного кадра, блокировка).")
                return False, 0

            # 2. Межкадровый детектор статичных водяных знаков (Temporal Persistence)
            static_watermark_detected = False
            if len(frames_list) >= 3 and self.cfg.get("check_watermarks", True):
                edge_maps = [cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 40, 120) for _, _, f in frames_list]
                mean_edges = np.mean(edge_maps, axis=0)

                # Статичные грани (присутствуют в 45-50%+ кадров в одной точке)
                static_mask = (mean_edges > 105).astype(np.uint8) * 255
                conn_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
                closed_static = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, conn_kernel)

                cnts, _ = cv2.findContours(closed_static, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                fh, fw = frames_list[0][2].shape[:2]
                top_h = int(fh * 0.22)
                bot_h = int(fh * 0.24)
                corner_w = int(fw * 0.28)

                for c in cnts:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    area = cw * ch
                    if area >= 45 and cw >= 12 and ch >= 8 and area <= (fw * fh * 0.40):
                        in_top_corner = (cy < top_h) and (cx < corner_w or cx + cw > fw - corner_w)
                        in_bot_corner = (cy + ch > fh - bot_h) and (cx < corner_w or cx + cw > fw - corner_w)
                        in_sub_band = (cy + ch > fh - bot_h)
                        in_top_band = (cy < top_h)
                        in_center = (top_h <= cy <= fh - bot_h)

                        zone_name = "углу"
                        if in_top_corner or in_bot_corner:
                            zone_name = "углу кадра"
                        elif in_sub_band:
                            zone_name = "нижней полосе"
                        elif in_top_band:
                            zone_name = "верхней полосе"
                        elif in_center:
                            zone_name = "центре кадра"

                        static_watermark_detected = True
                        peak_confidence = max(peak_confidence, 0.95)
                        desc = f"[STATIC LOGO] статичный водяной знак в {zone_name} ({cw}x{ch}, score=0.95)"
                        detected_reasons.append(desc)
                        self.app.log(f"[Watermark Debug] ⚡ ОБНАРУЖЕН СТАТИЧНЫЙ ВОДЯНОЙ ЗНАК: {desc}")

                        if in_center:
                            center_text_hits += 1
                            max_depth_seen = max(max_depth_seen, 0.5)
                        else:
                            depth = (top_h / fh) if "верх" in zone_name else (bot_h / fh)
                            max_depth_seen = max(max_depth_seen, depth)

            # Проверка ракурса сверху
            if top_down_only and top_down_hits == 0:
                self.app.log(f"[Watermark Debug] ✗ Отклонено: '{video_name}' — ракурс с дороги/земли (не вид сверху/дрон).")
                return False, 0

            wm_mode = self.cfg.get("watermark_mode", "Строгий фильтр").lower()

            # СТРОГИЙ РЕЖИМ (Скачивать ТОЛЬКО 100% чистые видео без водяных знаков)
            if "строгий" in wm_mode:
                if watermark_hits > 0 or static_watermark_detected:
                    summary_reason = "; ".join(list(dict.fromkeys(detected_reasons))[:2])
                    self.app.log(
                        f"[Watermark Debug] ИТОГ: REJECTED ✗ ('{video_name}': проверено {checked_frames_count} кадров, "
                        f"пиковый score={peak_confidence:.2f}, {summary_reason}). Пропуск ролика."
                    )
                    return False, 0
                self.app.log(
                    f"[Watermark Debug] ИТОГ: PASSED ✓ ('{video_name}': проверено {checked_frames_count} кадров, "
                    f"пиковый score={peak_confidence:.2f}, водяных знаков не обнаружено)."
                )
                return True, self.cfg.get("crop_percent", 0)

            # УМНЫЙ РЕЖИМ (Автообрезка краев + отсев центральных титров)
            elif "умный" in wm_mode:
                if center_text_hits >= 1:
                    self.app.log(
                        f"[Watermark Debug] ИТОГ: REJECTED ✗ ('{video_name}': в центре кадра обнаружены неустранимые титры/водяной знак, "
                        f"пиковый score={peak_confidence:.2f})."
                    )
                    return False, 0
                crop_pct = self.cfg.get("crop_percent", 0)
                if watermark_hits > 0 or static_watermark_detected:
                    adaptive_crop = min(25, int(max_depth_seen * 100) + 4)
                    crop_pct = max(crop_pct, adaptive_crop)
                    self.app.log(
                        f"[Watermark Debug] ИТОГ: PASSED (АВТООБРЕЗКА) ⚡ ('{video_name}': обнаружен краевой знак, "
                        f"адаптивный зум {crop_pct}%, пиковый score={peak_confidence:.2f})."
                    )
                else:
                    self.app.log(
                        f"[Watermark Debug] ИТОГ: PASSED ✓ ('{video_name}': чистый видеоряд, score={peak_confidence:.2f})."
                    )
                return True, crop_pct

            # ОТКЛЮЧЕНО
            self.app.log(f"[Watermark Debug] ИТОГ: PASSED ('{video_name}': фильтр водяных знаков отключен в настройках).")
            return True, self.cfg.get("crop_percent", 0)

        except Exception as e:
            self.app.log(f"[Watermark Debug] ✗ Критическая ошибка анализа видео '{video_name}': {e}. Принудительный вердикт: REJECTED.")
            return False, 0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _slice_and_filter_video(self, source_path: str, yolo_model: Optional[YOLO], needed_clips: int, top_down_only: bool,
                                video_title: str = "", video_desc: str = "", query: str = "") -> int:
        total_duration = self._get_video_duration(source_path)
        clip_len = self.cfg["clip_duration"]

        if total_duration < 4.0:
            self.app.log(f"Видео слишком короткое ({round(total_duration, 1)} сек < 4 сек). Пропуск.")
            return 0

        created_count = 0
        category_hint = self._classify_video_category(video_title, video_desc, query)

        # СЛУЧАЙ 1: Короткое видео или Shorts (длительность <= длины одного клипа)
        if total_duration <= clip_len:
            self.app.log(f"Видео формата Shorts / короткий ролик ({round(total_duration, 1)} сек). Обработка целиком...")
            self.app.set_status(f"Обработка клипа ({created_count + 1}/{needed_clips})...")

            raw_slice_path = str(self.temp_dir / f"short_slice_{int(time.time())}.mp4")

            ffmpeg_cmd = [
                self.cfg["ffmpeg"], "-y",
                "-i", source_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "aac",
                raw_slice_path
            ]
            res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(raw_slice_path):
                # Проверка ракурса и водяных знаков
                passed_wm, final_crop_pct = self._verify_clip_watermarks_and_angle(raw_slice_path, top_down_only)
                if not passed_wm:
                    try:
                        os.remove(raw_slice_path)
                    except Exception:
                        pass
                    return 0

                # ИИ-проверка
                passed_yolo = True
                yolo_counts = {"ground": 0, "air": 0}
                if self.cfg["use_ai_filter"] and yolo_model:
                    passed_yolo, yolo_counts = self._verify_traffic_with_yolo(raw_slice_path, yolo_model, category_hint)

                if passed_yolo:
                    category = self._classify_video_category(video_title, video_desc, query, yolo_counts)
                    self.app.log(f"✓ Проверка пройдена успешно. Категория: [{category.upper()}]. Экспорт в папку '{category}'...")
                    self._apply_antiwatermark_and_export(raw_slice_path, created_count + 1, final_crop_pct, category=category)
                    created_count += 1
                    progress = created_count / max(1, self.cfg["target_clips"])
                    self.app.set_status(f"Готово клипов: {created_count} из {self.cfg['target_clips']}", progress)
                else:
                    self.app.log("✗ Ролик отклонен ИИ (не обнаружено требуемой техники).")

                try:
                    if os.path.exists(raw_slice_path):
                        os.remove(raw_slice_path)
                except Exception:
                    pass

            return created_count

        # СЛУЧАЙ 2: Длинное видео (нарезка на фрагменты по clip_len секунд)
        self.app.log(f"Длинное видео: {int(total_duration)} сек. Нарезка на фрагменты по {clip_len} сек...")
        current_offset = 5.0

        while current_offset + clip_len <= total_duration:
            if created_count >= needed_clips or self.stop_event.is_set():
                break

            slice_idx = int(current_offset)
            raw_slice_path = str(self.temp_dir / f"raw_slice_{slice_idx}.mp4")

            self.app.log(f"Нарезка фрагмента c {int(current_offset)} по {int(current_offset + clip_len)} сек...")
            self.app.set_status(f"Нарезка клипа ({created_count + 1}/{needed_clips})...")

            ffmpeg_cmd = [
                self.cfg["ffmpeg"], "-y",
                "-ss", str(current_offset),
                "-i", source_path,
                "-t", str(clip_len),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "aac",
                "-avoid_negative_ts", "1",
                raw_slice_path
            ]

            res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0 or not os.path.exists(raw_slice_path):
                self.app.log(f"Ошибка нарезки на секунде {current_offset}. Переход к следующему сегменту...")
                current_offset += clip_len
                continue

            # 1. Проверка ракурса и водяных знаков на фрагменте
            passed_wm, final_crop_pct = self._verify_clip_watermarks_and_angle(raw_slice_path, top_down_only)
            if not passed_wm:
                current_offset += clip_len
                try:
                    os.remove(raw_slice_path)
                except Exception:
                    pass
                continue

            # 2. ИИ-детекция техники
            passed_yolo = True
            yolo_counts = {"ground": 0, "air": 0}
            if self.cfg["use_ai_filter"] and yolo_model:
                passed_yolo, yolo_counts = self._verify_traffic_with_yolo(raw_slice_path, yolo_model, category_hint)

            if passed_yolo:
                category = self._classify_video_category(video_title, video_desc, query, yolo_counts)
                self.app.log(f"✓ Кадры прошли проверку ИИ. Категория: [{category.upper()}]. Экспорт в папку '{category}'...")
                self._apply_antiwatermark_and_export(raw_slice_path, created_count + 1, final_crop_pct, category=category)
                created_count += 1
                progress = created_count / max(1, self.cfg["target_clips"])
                self.app.set_status(f"Готово клипов: {created_count} из {self.cfg['target_clips']}", progress)
            else:
                self.app.log("✗ Фрагмент отклонен (не обнаружено требуемой техники).")

            try:
                if os.path.exists(raw_slice_path):
                    os.remove(raw_slice_path)
            except Exception:
                pass

            current_offset += clip_len

        return created_count

    def _verify_traffic_with_yolo(self, video_path: str, model: YOLO, category_hint: str = "грузовики") -> Tuple[bool, Dict[str, int]]:
        """
        ИИ-детекция техники (YOLOv8):
        - Грузовики: фуры, тягачи, самосвалы, лесовозы (COCO class 7)
        - Наземная техника: танки, БМП, БТР, колонны, автомобили (COCO classes 2, 5, 6, 7)
        - Воздушная техника: дроны, БПЛА, квадрокоптеры, самолеты (COCO classes 4, 14, 38)
        - Разрешение imgsz=960 (находит мелкие объекты с высоты птичьего полета / дрона)
        """
        self.app.set_status(f"ИИ-анализ YOLOv8 ({category_hint})...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, {"truck": 0, "ground": 0, "air": 0}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        frame_area = frame_w * frame_h

        step_seconds = 1.5 if (total_frames / fps) <= 40 else 2.5
        step_frames = max(1, int(fps * step_seconds))

        sample_indices = list(range(0, total_frames, step_frames))
        if not sample_indices:
            cap.release()
            return False, {"truck": 0, "ground": 0, "air": 0}

        ai_mode = self.cfg.get("ai_mode", "").lower()
        strict_cargo = "грузом" in ai_mode or "строгий" in ai_mode
        drone_only_mode = "дроны и бпла" in ai_mode

        truck_frames = 0
        total_truck_seen = 0
        ground_frames = 0
        total_ground_seen = 0
        air_frames = 0
        total_air_seen = 0
        total_samples = 0

        for idx in sample_indices:
            if self.stop_event.is_set():
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            total_samples += 1

            results = model.predict(source=frame, verbose=False, conf=0.25, imgsz=960)

            frame_has_truck = False
            frame_has_ground = False
            frame_has_air = False

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    xyxy = boxes.xyxy[i].tolist()
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    box_area = bw * bh
                    aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

                    # 1. Грузовики (7)
                    is_truck = (cls_id == 7)
                    if is_truck and box_area > (frame_area * 0.001):
                        frame_has_truck = True
                        total_truck_seen += 1

                    # 2. Наземная техника: грузовики (7), автобусы (5), поезда (6), легковые/броневики/танки (2)
                    is_ground = (cls_id in (7, 5, 6)) or (cls_id == 2 and (aspect_ratio >= 1.25 or box_area > frame_area * 0.003))

                    if is_ground and box_area > (frame_area * 0.001):
                        frame_has_ground = True
                        total_ground_seen += 1

                    # 3. Воздушная техника: самолеты/БПЛА (4), птицы/воздушные силуэты (14, 38)
                    is_air = (cls_id == 4) or (cls_id in (14, 38) and box_area < frame_area * 0.08)
                    if is_air:
                        frame_has_air = True
                        total_air_seen += 1

            if frame_has_truck:
                truck_frames += 1
            if frame_has_ground:
                ground_frames += 1
            if frame_has_air:
                air_frames += 1

        cap.release()

        if total_samples == 0:
            return False, {"truck": 0, "ground": 0, "air": 0}

        yolo_counts = {"truck": total_truck_seen, "ground": total_ground_seen, "air": total_air_seen}

        # Решение о прохождении проверки в зависимости от категории
        if category_hint == "воздушная" or drone_only_mode:
            passed = True
            self.app.log(f"[YOLOv8] Воздушная техника (Дроны/БПЛА): подтверждено в {total_samples} кадрах (детектировано воздушных объектов: {total_air_seen}).")
        elif category_hint == "грузовики":
            truck_ratio = truck_frames / total_samples
            ground_ratio = ground_frames / total_samples
            if strict_cargo:
                passed = (truck_ratio >= 0.20 or ground_ratio >= 0.25) or (total_truck_seen >= 2 or total_ground_seen >= 2)
            else:
                passed = (truck_ratio >= 0.10 or ground_ratio >= 0.15) or (total_truck_seen >= 1 or total_ground_seen >= 1)
            pct = int(truck_ratio * 100)
            self.app.log(f"[YOLOv8] Грузовики: замечено {total_truck_seen} ед. (всего техники: {total_ground_seen}) в {truck_frames} из {total_samples} кадров ({pct}%).")
        else:
            ground_ratio = ground_frames / total_samples
            if strict_cargo:
                passed = (ground_ratio >= 0.25) or (total_ground_seen >= 2)
            else:
                passed = (ground_ratio >= 0.15) or (total_ground_seen >= 1)
            pct = int(ground_ratio * 100)
            self.app.log(f"[YOLOv8] Наземная техника: замечено {total_ground_seen} ед. в {ground_frames} из {total_samples} кадров ({pct}%).")

        return passed, yolo_counts

    def _apply_antiwatermark_and_export(self, input_clip: str, clip_number: int, crop_percent: Optional[int] = None, category: str = "грузовики"):
        """
        Обработка FFmpeg:
        - Адаптивный срез водяных знаков
        - Сохранение в подпапку 'грузовики', 'наземная' (военная техника) или 'воздушная' (дроны/БПЛА)
        - 9:16 (1080x1920) и 16:9 (1920x1080)
        - Удаление звука (-an) при необходимости
        """
        actual_crop = crop_percent if crop_percent is not None else self.cfg["crop_percent"]
        crop_pct = actual_crop / 100.0
        k = max(0.55, min(1.0, 1.0 - crop_pct))
        fmt_mode = self.cfg["format_mode"].lower()
        remove_audio = self.cfg["remove_audio"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if category == "воздушная":
            cat_clean = "воздушная"
            prefix = "drone"
        elif category == "грузовики":
            cat_clean = "грузовики"
            prefix = "truck"
        else:
            cat_clean = "наземная"
            prefix = "ground"

        base_cat_dir = self.out_dir / cat_clean
        base_cat_dir.mkdir(parents=True, exist_ok=True)

        prefix = "drone" if cat_clean == "воздушная" else ("truck" if cat_clean == "грузовики" else "ground")

        audio_flags = ["-an"] if remove_audio else ["-c:a", "aac", "-b:a", "192k"]

        w, h = self._get_video_dimensions(input_clip)
        is_vertical = h > w

        # 1. Экспорт 9:16 (Shorts/TikTok/Reels - 1080x1920)
        if "9:16" in fmt_mode or "оба" in fmt_mode:
            dest_dir = base_cat_dir / "9_16" if "оба" in fmt_mode else base_cat_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_file = str(dest_dir / f"{prefix}_9x16_{timestamp}_{clip_number}.mp4")

            self.app.log(f"Рендеринг 9:16 (1080x1920) в папку '{cat_clean}' с анти-вотермарк зумом {int(crop_pct*100)}%...")
            if is_vertical:
                vf_9_16 = f"crop=iw*{k}:ih*{k}:(iw-ow)/2:(ih-oh)/2,scale=1080:1920:flags=lanczos,setsar=1"
            else:
                vf_9_16 = f"crop=ih*{k}*9/16:ih*{k}:(iw-ow)/2:(ih-oh)/2,scale=1080:1920:flags=lanczos,setsar=1"

            cmd_9_16 = [
                self.cfg["ffmpeg"], "-y",
                "-i", input_clip,
                "-vf", vf_9_16,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p"
            ] + audio_flags + [out_file]

            subprocess.run(cmd_9_16, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(out_file):
                self.app.log(f"✓ Сохранен клип [{cat_clean.upper()}] 9:16: {os.path.basename(out_file)} (папка: '{cat_clean}')")

        # 2. Экспорт 16:9 (YouTube - 1920x1080)
        if "16:9" in fmt_mode or "оба" in fmt_mode:
            dest_dir = base_cat_dir / "16_9" if "оба" in fmt_mode else base_cat_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_file = str(dest_dir / f"{prefix}_16x9_{timestamp}_{clip_number}.mp4")

            self.app.log(f"Рендеринг 16:9 (1920x1080) в папку '{cat_clean}' с зумом {int(crop_pct*100)}%...")
            if is_vertical:
                vf_16_9 = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            else:
                vf_16_9 = f"crop=iw*{k}:ih*{k}:(iw-ow)/2:(ih-oh)/2,scale=1920:1080:flags=lanczos,setsar=1"

            cmd_16_9 = [
                self.cfg["ffmpeg"], "-y",
                "-i", input_clip,
                "-vf", vf_16_9,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p"
            ] + audio_flags + [out_file]

            subprocess.run(cmd_16_9, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(out_file):
                self.app.log(f"✓ Сохранен клип [{cat_clean.upper()}] 16:9: {os.path.basename(out_file)} (папка: '{cat_clean}')")

    def _cleanup_temp(self):
        try:
            for item in self.temp_dir.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
            self.app.log("Временные файлы очищены.")
        except Exception as e:
            self.app.log(f"Очистка временных файлов: {e}")


def main():
    app = TruckVideoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
