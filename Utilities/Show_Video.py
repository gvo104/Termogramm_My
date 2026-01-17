import cv2
from typing import List, Dict, Tuple, Union
import numpy as np


def show_video_realtime(frames: Union[np.ndarray, List[np.ndarray]], 
                       fps: int = 25, 
                       window_name: str = "Video Viewer",
                       colormap: int = cv2.COLORMAP_JET,
                       normalize: bool = True,
                       global_normalization: bool = True):
    """
    Показывает numpy массив кадров в виде видео с возможностью пролистывания
    
    Parameters:
    -----------
    frames : np.ndarray или list of np.ndarray
        Массив кадров формы (N, H, W) или список массивов (H, W)
    fps : int
        Количество кадров в секунду
    window_name : str
        Название окна
    colormap : int
        OpenCV colormap (cv2.COLORMAP_*)
    normalize : bool
        Нормализовать ли изображение для отображения
    global_normalization : bool
        Использовать глобальную нормализацию по всем кадрам
    """
    # Преобразуем в список numpy массивов если нужно
    if isinstance(frames, np.ndarray):
        if frames.ndim == 3:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        else:
            raise ValueError("Input array must be 3D (frames, height, width)")
    else:
        frame_list = frames
    
    total_frames = len(frame_list)
    delay = int(1000 / fps)
    current_frame = 0
    paused = False
    
    # Предварительная нормализация если включена глобальная
    if normalize and global_normalization:
        # Собираем все значения для глобальной нормализации
        all_values = np.concatenate([f.flatten() for f in frame_list])
        min_val = np.percentile(all_values, 1)
        max_val = np.percentile(all_values, 99)
        max_val = max(max_val, min_val + 1e-6)
        print(f"Глобальная нормализация: min={min_val:.2f}, max={max_val:.2f}")
    else:
        min_val = max_val = None
    
    # Создаем окно
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 750, 750)
    
    # Функция для подготовки кадра к отображению
    def prepare_frame(frame_idx: int) -> np.ndarray:
        frame = frame_list[frame_idx]
        
        if normalize:
            if global_normalization:
                # Используем предварительно вычисленные глобальные значения
                frame_clip = np.clip(frame, min_val, max_val)
                frame_norm = ((frame_clip - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # Локальная нормализация для каждого кадра
                frame_min = np.percentile(frame, 1)
                frame_max = np.percentile(frame, 99)
                frame_max = max(frame_max, frame_min + 1e-6)
                frame_clip = np.clip(frame, frame_min, frame_max)
                frame_norm = ((frame_clip - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        else:
            # Просто приводим к uint8
            frame_norm = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Применяем colormap если изображение одноканальное
        if len(frame_norm.shape) == 2 or (len(frame_norm.shape) == 3 and frame_norm.shape[2] == 1):
            frame_display = cv2.applyColorMap(frame_norm, colormap)
        else:
            frame_display = frame_norm
        
        return frame_display
    
    print("Управление:")
    print("  SPACE - пауза/продолжить")
    print("  A/←    - предыдущий кадр")
    print("  D/→    - следующий кадр")
    print("  W/↑    - +10 кадров")
    print("  S/↓    - -10 кадров")
    print("  Home   - первый кадр")
    print("  End    - последний кадр")
    print("  R      - сбросить воспроизведение")
    print("  +/-    - изменить скорость (FPS)")
    print("  C      - переключить colormap")
    print("  N      - переключить нормализацию")
    print("  G      - переключить глобальную/локальную нормализацию")
    print("  ESC    - выход")
    
    # Доступные colormaps
    available_colormaps = [
        cv2.COLORMAP_JET,
        cv2.COLORMAP_HOT,
        cv2.COLORMAP_COOL,
        cv2.COLORMAP_SPRING,
        cv2.COLORMAP_SUMMER,
        cv2.COLORMAP_AUTUMN,
        cv2.COLORMAP_WINTER,
        cv2.COLORMAP_BONE,
        cv2.COLORMAP_OCEAN,
        cv2.COLORMAP_RAINBOW
    ]
    current_colormap_idx = 0
    colormap_names = ["JET", "HOT", "COOL", "SPRING", "SUMMER", "AUTUMN", "WINTER", "BONE", "OCEAN", "RAINBOW"]
    
    while True:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        
        # Подготавливаем и отображаем текущий кадр
        frame_display = prepare_frame(current_frame)
        
        # Добавляем информацию на кадр
        text_lines = [
            f"Frame: {current_frame + 1}/{total_frames}",
            f"FPS: {fps} {'[PAUSED]' if paused else ''}",
            f"Colormap: {colormap_names[current_colormap_idx]}",
            f"Norm: {'Global' if global_normalization else 'Local'}",
            f"Size: {frame_list[current_frame].shape}"
        ]
        
        for i, text in enumerate(text_lines):
            y_position = 30 + i * 25
            cv2.putText(frame_display, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # Тень текста для лучшей читаемости
            cv2.putText(frame_display, text, (11, y_position + 1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Прогресс-бар
        bar_width = frame_display.shape[1] - 20
        bar_height = 10
        bar_x, bar_y = 10, frame_display.shape[0] - 20
        progress = (current_frame + 1) / total_frames
        
        # Фон прогресс-бара
        cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Заполненная часть
        cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        # Рамка
        cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        
        cv2.imshow(window_name, frame_display)
        
        # Обработка клавиш
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Пробел - пауза
            paused = not paused
        elif key in [ord('a'), ord('A'), 81]:  # Назад (A или ←)
            current_frame = max(0, current_frame - 1)
            paused = True
        elif key in [ord('d'), ord('D'), 83]:  # Вперёд (D или →)
            current_frame = min(total_frames - 1, current_frame + 1)
            paused = True
        elif key in [ord('w'), ord('W'), 82]:  # +10 кадров (W или ↑)
            current_frame = min(total_frames - 1, current_frame + 10)
            paused = True
        elif key in [ord('s'), ord('S'), 84]:  # -10 кадров (S или ↓)
            current_frame = max(0, current_frame - 10)
            paused = True
        elif key == 80:  # Home - первый кадр
            current_frame = 0
            paused = True
        elif key == 87:  # End - последний кадр
            current_frame = total_frames - 1
            paused = True
        elif key == ord('r'):  # R - сброс
            current_frame = 0
            paused = False
        elif key == ord('+'):  # + - увеличить FPS
            fps = min(60, fps + 5)
            delay = int(1000 / fps)
        elif key == ord('-'):  # - - уменьшить FPS
            fps = max(1, fps - 5)
            delay = int(1000 / fps)
        elif key == ord('c'):  # C - сменить colormap
            current_colormap_idx = (current_colormap_idx + 1) % len(available_colormaps)
            colormap = available_colormaps[current_colormap_idx]
        elif key == ord('n'):  # N - переключить нормализацию
            normalize = not normalize
        elif key == ord('g'):  # G - переключить глобальную/локальную нормализацию
            global_normalization = not global_normalization
            if normalize and global_normalization:
                # Пересчитываем глобальные значения
                all_values = np.concatenate([f.flatten() for f in frame_list])
                min_val = np.percentile(all_values, 1)
                max_val = np.percentile(all_values, 99)
                max_val = max(max_val, min_val + 1e-6)
        elif not paused:
            current_frame = (current_frame + 1) % total_frames
    
    cv2.destroyAllWindows()
    print("Просмотр завершен")








def draw_triangle(img, cx, cy, size, color):
    """
    Простой равносторонний треугольник по центру (cx, cy)
    """
    cx, cy = int(cx), int(cy)
    h = size

    pts = np.array([
        [cx,        cy - h],
        [cx - h//2, cy + h//2],
        [cx + h//2, cy + h//2]
    ], np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)








def get_corridor_segment(first_center, end_center, corridor_width=20):
    """
    Возвращает координаты четырех углов сегмента коридора от first_center до end_center.
    """
    vec = end_center - first_center
    length = np.linalg.norm(vec)
    if length < 1e-6:
        direction = np.array([0, 1], dtype=float)
    else:
        direction = vec / length
    perp = np.array([-direction[1], direction[0]])
    half_w = corridor_width / 2

    corners = np.array([
        first_center + perp * half_w,
        first_center - perp * half_w,
        end_center - perp * half_w,
        end_center + perp * half_w
    ], dtype=int)
    return corners






def show_video_with_detections(frames: np.ndarray,
                               cleaned_spatters: list,
                               static_spatters: list,
                               welding_boxes: list,
                               welding_ellipses: list,
                               hot_stripes_list: list = None,  # полосы для всех кадров
                               fps: int = 25,
                               window_name: str = "Welding Analysis",
                               colormap: int = cv2.COLORMAP_JET,
                               normalize: bool = True,
                               global_normalization: bool = True,
                               dead_zone_radius: int = 50,
                               corridor_width: int = 20,
                               lookahead: int = 5):
    '''
    Главная функция для просмотра видео с нанесенными метками детекции
    '''

    if frames.ndim != 3:
        raise ValueError("Input array must be 3D")

    frame_list = [frames[i] for i in range(frames.shape[0])]
    total_frames = len(frame_list)
    delay = int(1000 / fps)
    current_frame = 0
    paused = False

    available_colormaps = [
        cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL,
        cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS
    ]
    current_colormap_idx = available_colormaps.index(colormap) if colormap in available_colormaps else 0
    colormap = available_colormaps[current_colormap_idx]

    if normalize and global_normalization:
        all_values = np.concatenate([f.flatten() for f in frame_list])
        min_val = np.percentile(all_values, 1)
        max_val = np.percentile(all_values, 99)
        max_val = max(max_val, min_val + 1e-6)
    else:
        min_val = max_val = None

    # ----------------------------
    # FLAGS TOGGLERS
    # ----------------------------
    show_box = False
    show_ellipse = True
    show_static = False
    show_corridor = False
    show_hot_stripes = True  # клавиша 5 для отображения горячих полос

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 900)
    cv2.imshow(window_name, np.zeros((20, 20), dtype=np.uint8))
    cv2.waitKey(60)

    def prepare_frame_with_detections(frame_idx: int) -> np.ndarray:
        frame = frame_list[frame_idx]

        # --- нормализация ---
        if normalize:
            if global_normalization:
                frame_clip = np.clip(frame, min_val, max_val)
                frame_norm = ((frame_clip - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                f_min = np.percentile(frame, 1)
                f_max = np.percentile(frame, 99)
                f_max = max(f_max, f_min + 1e-6)
                frame_clip = np.clip(frame, f_min, f_max)
                frame_norm = ((frame_clip - f_min) / (f_max - f_min) * 255).astype(np.uint8)
        else:
            frame_norm = np.clip(frame, 0, 255).astype(np.uint8)

        frame_display = cv2.applyColorMap(frame_norm, colormap)

        # ----------------------------
        # Рисуем горячие полосы через эллипсы
        # ----------------------------
        if hot_stripes_list is not None and show_hot_stripes and hot_stripes_list[frame_idx]:
            for stripe in hot_stripes_list[frame_idx]:
                cx, cy, MAJOR, MINOR, angle = stripe
                if MAJOR < 2 or MINOR < 2:
                    continue
                center = (int(cx), int(cy))
                axes = (max(int(MAJOR / 2), 1), max(int(MINOR / 2), 1))
                angle_deg = np.degrees(angle) % 360
                cv2.ellipse(frame_display, center, axes, angle_deg, 0, 360, (0, 255, 255), 2)

        # ----------------------------
        # накопительный мертвый коридор
        # ----------------------------
        if show_corridor:
            corridor_segments = []
            first_center = None
            for idx in range(frame_idx + 1):
                ell = welding_ellipses[idx]
                if ell is not None:
                    center = np.array(ell[0], dtype=float)
                    if first_center is not None:
                        segment = get_corridor_segment(first_center, center, corridor_width)
                        corridor_segments.append(segment)
                    first_center = center

            for seg in corridor_segments:
                cv2.polylines(frame_display, [seg], isClosed=True, color=( 255, 0, 255), thickness=2)

        # ----------------------------
        # зона сварки
        # ----------------------------
        weld_box = welding_boxes[frame_idx]
        if weld_box is not None:
            if show_box:
                cx, cy, w, h = map(int, weld_box)
                x1, y1 = cx - w // 2, cy - h // 2
                x2, y2 = cx + w // 2, cy + h // 2
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if show_ellipse and welding_ellipses[frame_idx] is not None:
                try:
                    cv2.ellipse(frame_display, welding_ellipses[frame_idx], (0, 255, 0), 2)
                    (xc, yc), (_, _), _ = welding_ellipses[frame_idx]
                    cv2.circle(frame_display, (int(xc), int(yc)), dead_zone_radius, (0, 255, 255), 1)
                except:
                    pass

        # ----------------------------
        # обычные брызги
        # ----------------------------
        for sp in cleaned_spatters[frame_idx]:
            cx, cy, w, h = sp.astype(int)
            r = max(w, h) // 2 + 6
            r = max(r, 3)
            cv2.circle(frame_display, (cx, cy), r, (0, 0, 255), 1)

        # ----------------------------
        # статичные брызги
        # ----------------------------
        if show_static:
            for sp in static_spatters[frame_idx]:
                cx, cy, w, h = sp.astype(int)
                r = max(w, h) // 2 + 6
                pts = np.array([
                    [cx, cy - r],
                    [cx - r, cy + r],
                    [cx + r, cy + r]
                ], np.int32)
                cv2.polylines(frame_display, [pts], True, (255, 255, 0), 2)

        # # текст + прогресс
        # text = f"Frame {frame_idx+1}/{total_frames} | FPS {fps}"
        # cv2.putText(frame_display, text, (10, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # bar_w = int((frame_idx + 1) / total_frames * frame_display.shape[1])
        # cv2.rectangle(frame_display,
        #               (0, frame_display.shape[0] - 12),
        #               (bar_w, frame_display.shape[0]),
        #               (255, 255, 255), -1)

        return frame_display

    # ---------------------------------------
    #                  MAIN LOOP
    # ---------------------------------------
    while True:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        disp = prepare_frame_with_detections(current_frame)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(0 if paused else delay) & 0xFF

        # выход
        if key == 27:
            break
        # пауза
        elif key == ord(' '):
            paused = not paused
        # навигация
        elif key in [ord('a'), ord('A'), 81]:
            current_frame = max(0, current_frame - 1); paused = True
        elif key in [ord('d'), ord('D'), 83]:
            current_frame = min(total_frames - 1, current_frame + 1); paused = True
        elif key in [ord('w'), ord('W'), 82]:
            current_frame = min(total_frames - 1, current_frame + 10); paused = True
        elif key in [ord('s'), ord('S'), 84]:
            current_frame = max(0, current_frame - 10); paused = True
        # начало/конец
        elif key == 80: current_frame = 0; paused = True
        elif key == 87: current_frame = total_frames - 1; paused = True
        # refresh
        elif key == ord('r'):
            current_frame = 0; paused = False
        # FPS
        elif key == ord('+'):
            fps = min(60, fps + 5); delay = int(1000 / fps)
        elif key == ord('-'):
            fps = max(1, fps - 5); delay = int(1000 / fps)
        # colormap
        elif key == ord('c'):
            current_colormap_idx = (current_colormap_idx + 1) % len(available_colormaps)
            colormap = available_colormaps[current_colormap_idx]
        # normalization
        elif key == ord('n'):
            normalize = not normalize
        elif key == ord('g'):
            global_normalization = not global_normalization
            if normalize and global_normalization:
                all_values = np.concatenate([f.flatten() for f in frame_list])
                min_val = np.percentile(all_values, 1)
                max_val = np.percentile(all_values, 99)
                max_val = max(max_val, min_val + 1e-6)
        # TOGGLES (1–5)
        elif key == ord('1'): show_box = not show_box
        elif key == ord('2'): show_ellipse = not show_ellipse
        elif key == ord('3'): show_static = not show_static
        elif key == ord('4'): show_corridor = not show_corridor
        elif key == ord('5'): show_hot_stripes = not show_hot_stripes
        # автоплей
        elif not paused:
            current_frame = (current_frame + 1) % total_frames

    cv2.destroyAllWindows()
    print("Просмотр завершен")


def show_video_with_tracking(frames: np.ndarray,
                           cleaned_spatters_id: list,
                           welding_boxes: list = None,
                           welding_ellipses: list = None,
                           fps: int = 25,
                           window_name: str = "Spatter Tracking",
                           colormap: int = cv2.COLORMAP_JET,
                           normalize: bool = True,
                           global_normalization: bool = True,
                           trail_length: int = 10,
                           show_trails: bool = True):
    """
    Визуализация трекинга брызг с цветами по ID и траекториями
    
    Args:
        frames: numpy array of shape (N, H, W) - video frames
        cleaned_spatters_id: список массивов [x, y, w, h, id] для каждого кадра
        welding_boxes: список боксов зоны сварки (опционально)
        welding_ellipses: список эллипсов зоны сварки (опционально)
        fps: frames per second
        window_name: имя окна
        colormap: цветовая карта OpenCV
        normalize: нормализация температуры
        global_normalization: глобальная или локальная нормализация
        trail_length: длина траектории (количество предыдущих кадров)
        show_trails: показывать траектории движения
    """
    
    if frames.ndim != 3:
        raise ValueError("Input array must be 3D")

    frame_list = [frames[i] for i in range(frames.shape[0])]
    total_frames = len(frame_list)
    delay = int(1000 / fps)
    current_frame = 0
    paused = False

    available_colormaps = [
        cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_COOL,
        cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS
    ]
    current_colormap_idx = available_colormaps.index(colormap) if colormap in available_colormaps else 0
    colormap = available_colormaps[current_colormap_idx]

    if normalize and global_normalization:
        all_values = np.concatenate([f.flatten() for f in frame_list])
        min_val = np.percentile(all_values, 1)
        max_val = np.percentile(all_values, 99)
        max_val = max(max_val, min_val + 1e-6)
    else:
        min_val = max_val = None

    # Словарь для хранения истории позиций по ID
    position_history = {}
    
    # Генерация цветовой палитры для ID
    def get_color_for_id(spatter_id):
        np.random.seed(spatter_id % 1000)  # для воспроизводимости
        return tuple(np.random.randint(0, 255, 3).tolist())

    # ----------------------------
    # FLAGS TOGGLERS (из основной функции)
    # ----------------------------
    show_box = False
    show_ellipse = True
    show_corridor = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 900)

    def prepare_frame_with_tracking(frame_idx: int) -> np.ndarray:
        frame = frame_list[frame_idx]

        # Нормализация
        if normalize:
            if global_normalization:
                frame_clip = np.clip(frame, min_val, max_val)
                frame_norm = ((frame_clip - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                f_min = np.percentile(frame, 1)
                f_max = np.percentile(frame, 99)
                f_max = max(f_max, f_min + 1e-6)
                frame_clip = np.clip(frame, f_min, f_max)
                frame_norm = ((frame_clip - f_min) / (f_max - f_min) * 255).astype(np.uint8)
        else:
            frame_norm = np.clip(frame, 0, 255).astype(np.uint8)

        frame_display = cv2.applyColorMap(frame_norm, colormap)

        # Обновляем историю позиций
        current_spatters = cleaned_spatters_id[frame_idx]
        
        # Удаляем старые записи из истории
        current_ids = set()
        if len(current_spatters) > 0:
            for spatter in current_spatters:
                spatter_id = int(spatter[4])
                current_ids.add(spatter_id)
                x, y = int(spatter[0]), int(spatter[1])
                
                # Добавляем текущую позицию в историю
                if spatter_id not in position_history:
                    position_history[spatter_id] = []
                position_history[spatter_id].append((frame_idx, x, y))
                
                # Ограничиваем длину истории
                if len(position_history[spatter_id]) > trail_length:
                    position_history[spatter_id] = position_history[spatter_id][-trail_length:]
        
        # Очищаем историю для ID которых больше нет
        ids_to_remove = []
        for spatter_id in position_history:
            if spatter_id not in current_ids:
                # Проверяем, не устарела ли история
                if position_history[spatter_id]:
                    last_frame, _, _ = position_history[spatter_id][-1]
                    if frame_idx - last_frame > trail_length:
                        ids_to_remove.append(spatter_id)
        
        for spatter_id in ids_to_remove:
            del position_history[spatter_id]

        # Рисуем траектории
        if show_trails:
            for spatter_id, history in position_history.items():
                if len(history) > 1:
                    color = get_color_for_id(spatter_id)
                    # Рисуем линии между точками истории
                    points = []
                    for hist_frame_idx, x, y in history:
                        # Пропускаем точки из будущих кадров (на всякий случай)
                        if hist_frame_idx <= frame_idx:
                            points.append((x, y))
                    
                    if len(points) >= 2:
                        # Рисуем плавную траекторию
                        for i in range(1, len(points)):
                            alpha = i / len(points)  # прозрачность для старых точек
                            thickness = max(1, int(3 * alpha))
                            cv2.line(frame_display, points[i-1], points[i], color, thickness)
                        
                        # Рисуем точки на траектории
                        for i, (x, y) in enumerate(points):
                            alpha = i / len(points)
                            radius = max(1, int(3 * alpha))
                            cv2.circle(frame_display, (x, y), radius, color, -1)

        # Рисуем текущие брызги
        if len(current_spatters) > 0:
            for spatter in current_spatters:
                x, y, w, h, spatter_id = spatter
                x, y = int(x), int(y)
                spatter_id = int(spatter_id)
                
                color = get_color_for_id(spatter_id)
                
                # Рисуем круг для брызги
                radius = max(3, int(max(w, h) // 2 + 2))
                cv2.circle(frame_display, (x, y), radius, color, 2)
                
                # Рисуем крестик в центре
                cross_size = radius + 2
                cv2.line(frame_display, (x-cross_size, y), (x+cross_size, y), color, 1)
                cv2.line(frame_display, (x, y-cross_size), (x, y+cross_size), color, 1)

        # Рисуем зону сварки (если есть)
        if welding_ellipses is not None and welding_ellipses[frame_idx] is not None and show_ellipse:
            try:
                cv2.ellipse(frame_display, welding_ellipses[frame_idx], (0, 255, 0), 2)
            except:
                pass

        if welding_boxes is not None and welding_boxes[frame_idx] is not None and show_box:
            try:
                cx, cy, w, h = map(int, welding_boxes[frame_idx])
                x1, y1 = cx - w // 2, cy - h // 2
                x2, y2 = cx + w // 2, cy + h // 2
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 1)
            except:
                pass

        # # Только номер кадра и прогресс-бар (без статистики)
        # text = f"Frame {frame_idx+1}/{total_frames} | FPS {fps}"
        # cv2.putText(frame_display, text, (10, 25),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # # Прогресс-бар
        # bar_w = int((frame_idx + 1) / total_frames * frame_display.shape[1])
        # cv2.rectangle(frame_display,
        #              (0, frame_display.shape[0] - 12),
        #              (bar_w, frame_display.shape[0]),
        #              (255, 255, 255), -1)

        return frame_display

    # Управление
    while True:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        disp = prepare_frame_with_tracking(current_frame)
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(0 if paused else delay) & 0xFF

        # Управление (аналогично оригинальной функции)
        if key == 27: break  # ESC
        elif key == ord(' '): paused = not paused
        elif key in [ord('a'), ord('A'), 81]: current_frame = max(0, current_frame - 1); paused = True
        elif key in [ord('d'), ord('D'), 83]: current_frame = min(total_frames - 1, current_frame + 1); paused = True
        elif key in [ord('w'), ord('W'), 82]: current_frame = min(total_frames - 1, current_frame + 10); paused = True
        elif key in [ord('s'), ord('S'), 84]: current_frame = max(0, current_frame - 10); paused = True
        elif key == 80: current_frame = 0; paused = True
        elif key == 87: current_frame = total_frames - 1; paused = True
        elif key == ord('r'): current_frame = 0; paused = False
        elif key == ord('+'): fps = min(60, fps + 5); delay = int(1000 / fps)
        elif key == ord('-'): fps = max(1, fps - 5); delay = int(1000 / fps)
        elif key == ord('c'): current_colormap_idx = (current_colormap_idx + 1) % len(available_colormaps); colormap = available_colormaps[current_colormap_idx]
        elif key == ord('n'): normalize = not normalize
        elif key == ord('g'): global_normalization = not global_normalization
        # Флаги из основной функции
        elif key == ord('1'): show_box = not show_box
        elif key == ord('2'): show_ellipse = not show_ellipse
        elif key == ord('4'): show_corridor = not show_corridor
        # Управление трекингом
        elif key == ord('t'): show_trails = not show_trails  # вкл/выкл траектории
        elif key == ord(']'): trail_length = min(50, trail_length + 1)  # увеличить длину траектории
        elif key == ord('['): trail_length = max(1, trail_length - 1)  # уменьшить длину траектории
        # Автоплей
        elif not paused: current_frame = (current_frame + 1) % total_frames

    cv2.destroyAllWindows()
    print("Просмотр трекинга завершен")