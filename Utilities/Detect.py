import cv2
import numpy as np
from tqdm import tqdm
from math import radians, degrees, pi



from Utilities.Filter_Min_Loc_Log import min_loc_LoG
from Utilities.Show_Video import get_corridor_segment

def detect_spatters(frame, 
                    sigma=1.8, 
                    k_size=9, 
                    threshold=11, 
                    min_size=3,
                    max_size=20):
    """Детекция брызг с настраиваемыми параметрами"""
    filtered = min_loc_LoG(frame, k_size, sigma)
    filtered = ((filtered > threshold) * 255).astype(np.uint8)
    c, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.array([cv2.boundingRect(c[i])[0],cv2.boundingRect(c[i])[1],cv2.boundingRect(c[i])[0]+cv2.boundingRect(c[i])[2],cv2.boundingRect(c[i])[1]+cv2.boundingRect(c[i])[3]]) for i,_ in enumerate(c)]
    contours = np.array(contours)
    if not len(contours):
        return ()
    wh = contours[:, 2:4] - contours[:, :2]
    contours[:, :2] = contours[:, :2] + wh / 2
    contours[:, 2:4] = wh
    
    # ПРОСТЫЕ ОГРАНИЧЕНИЯ НА РАЗМЕР
    # Фильтруем по ширине и высоте
    width_ok = (wh[:, 0] >= min_size) & (wh[:, 0] <= max_size)
    height_ok = (wh[:, 1] >= min_size) & (wh[:, 1] <= max_size)
    size_filter = width_ok & height_ok
    
    contours = contours[size_filter]
    return contours

def detect_welding_zone(frame: np.ndarray,
                        threshold: int = 9.9,
                        min_area: int = 0.1,
                        max_fraction: float = 0.15):
    """
    Возвращает (box, ellipse).
    box = [x1, y1, x2, y2] или None
    ellipse = (center, axes, angle) или None

    max_fraction — максимальная доля кадра по ширине и высоте.
    Если зона превышает эту долю — считается, что её нет.
    """

    img = frame.astype(np.float32)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(img_norm, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Ограничение по доле кадра
    max_w = frame.shape[1] * max_fraction
    max_h = frame.shape[0] * max_fraction
    if w > max_w or h > max_h:
        return None, None

    box = np.array([x, y, x + w, y + h])

    ellipse = None
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)

    return box, ellipse

    


def detect_spatters_video(frames: np.ndarray, show_progress: bool = True) -> list:
    """
    Detect spatters in video sequence using detect_spatters function
    
    Args:
        frames: numpy array of shape (N, H, W) - video frames
        show_progress: whether to show progress bar
    
    Returns:
        List of spatter contours for each frame
    """
    all_spatters = []
    
    iterator = range(frames.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="Detecting spatters")
    
    for i in iterator:
        spatters = detect_spatters(frames[i])
        all_spatters.append(spatters)
    
    return all_spatters


def detect_welding_zone_video(frames: np.ndarray, 
                             show_progress: bool = True,
                             max_centers_history: int = 10,
                             max_velocity_history: int = 70,
                             min_centers_for_prediction: int = 5):
    """
    Детектирует зону сварки в видео с предсказанием при отсутствии детекции
    
    Аргументы:
        frames: numpy массив формы (N, H, W) - кадры видео
        show_progress: показывать ли прогресс-бар
        max_centers_history: максимальное количество центров для хранения в истории
        max_velocity_history: максимальное количество значений скорости для хранения
        min_centers_for_prediction: минимальное количество центров необходимое для предсказания
    """
    all_boxes = []
    all_ellipses = []
    
    welding_centers_history = []
    frame_indices_history = []
    velocity_history = []
    
    iterator = range(frames.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="Детекция зон сварки с предсказанием")

    for i in iterator:
        try:
            box, ellipse = detect_welding_zone(frames[i])
            
            if box is not None and ellipse is not None:
                # Зона сварки обнаружена
                all_boxes.append(box)
                all_ellipses.append(ellipse)
                
                # Сохраняем центр для истории
                center_x, center_y = ellipse[0]
                welding_centers_history.append((center_x, center_y))
                frame_indices_history.append(i)
                
                # Обновляем историю скоростей
                if len(welding_centers_history) >= 2:
                    last_center = welding_centers_history[-2]
                    current_center = welding_centers_history[-1]
                    frame_diff = frame_indices_history[-1] - frame_indices_history[-2]
                    
                    if frame_diff > 0:
                        velocity_x = (current_center[0] - last_center[0]) / frame_diff
                        velocity_y = (current_center[1] - last_center[1]) / frame_diff
                        velocity_history.append((velocity_x, velocity_y))
                
                # Ограничиваем размер истории
                if len(welding_centers_history) > max_centers_history:
                    welding_centers_history.pop(0)
                    frame_indices_history.pop(0)
                if len(velocity_history) > max_velocity_history:
                    velocity_history.pop(0)
                    
            else:
                # Зона сварки не обнаружена - предсказываем
                last_successful_box = None
                last_successful_ellipse = None
                
                if welding_centers_history:
                    last_successful_box = all_boxes[frame_indices_history[-1]]
                    last_successful_ellipse = all_ellipses[frame_indices_history[-1]]
                
                predicted_box, predicted_ellipse = predict_welding_zone(
                    welding_centers_history, 
                    velocity_history,
                    frame_indices_history,
                    i,
                    last_successful_box,
                    last_successful_ellipse,
                    min_centers_for_prediction
                )
                
                all_boxes.append(predicted_box)
                all_ellipses.append(predicted_ellipse)
                
        except Exception:
            # В случае ошибки предсказываем
            last_successful_box = None
            last_successful_ellipse = None
            
            if welding_centers_history:
                last_successful_box = all_boxes[frame_indices_history[-1]]
                last_successful_ellipse = all_ellipses[frame_indices_history[-1]]
            
            predicted_box, predicted_ellipse = predict_welding_zone(
                welding_centers_history, 
                velocity_history,
                frame_indices_history,
                i,
                last_successful_box,
                last_successful_ellipse,
                min_centers_for_prediction
            )
            
            all_boxes.append(predicted_box)
            all_ellipses.append(predicted_ellipse)

    return all_boxes, all_ellipses


def predict_welding_zone(centers_history, velocity_history, frame_indices_history, 
                        current_frame_idx, last_successful_box, last_successful_ellipse,
                        min_centers_for_prediction: int = 2):
    """
    Предсказывает положение зоны сварки на основе истории
    
    Аргументы:
        centers_history: список предыдущих центров [(x, y), ...]
        velocity_history: список скоростей [(vx, vy), ...]
        frame_indices_history: список номеров кадров с детекцией
        current_frame_idx: текущий номер кадра
        last_successful_box: последний успешно обнаруженный bounding box
        last_successful_ellipse: последний успешно обнаруженный эллипс
        min_centers_for_prediction: минимальное количество центров для предсказания
    """
    
    if len(centers_history) < min_centers_for_prediction or len(velocity_history) == 0:
        return last_successful_box, last_successful_ellipse
    
    # Вычисляем среднюю скорость
    avg_velocity_x = np.mean([v[0] for v in velocity_history])
    avg_velocity_y = np.mean([v[1] for v in velocity_history])
    
    # Берем последний известный центр
    last_center = centers_history[-1]
    last_frame_with_detection = frame_indices_history[-1]
    
    # Вычисляем смещение
    frames_since_last_detection = current_frame_idx - last_frame_with_detection
    displacement_x = avg_velocity_x * frames_since_last_detection
    displacement_y = avg_velocity_y * frames_since_last_detection
    
    # Новый предсказанный центр
    predicted_center_x = last_center[0] + displacement_x
    predicted_center_y = last_center[1] + displacement_y
    
    if last_successful_box is not None:
        # Предсказываем новый bounding box
        box_width = last_successful_box[2]
        box_height = last_successful_box[3]
        
        predicted_box = np.array([
            predicted_center_x - box_width / 2,
            predicted_center_y - box_height / 2,
            box_width,
            box_height
        ])
    else:
        predicted_box = None
    
    if last_successful_ellipse is not None:
        # Предсказываем новый эллипс
        (_, _), (major_axis, minor_axis), angle = last_successful_ellipse
        predicted_ellipse = ((predicted_center_x, predicted_center_y), 
                           (major_axis, minor_axis), angle)
    else:
        predicted_ellipse = None
    
    return predicted_box, predicted_ellipse


def remove_static_spatters(spatters_list, max_shift=7, min_frames=4):
    """
    Находит статичные брызги, которые остаются в окрестности max_shift
    более чем min_frames подряд.
    Возвращает:
        cleaned_spatters — динамические
        static_spatters  — статичные
    """

    # --- ПРИВЕДЕНИЕ К НОРМАЛЬНОМУ ФОРМАТУ ---
    if isinstance(spatters_list, tuple):
        spatters_list = list(spatters_list)

    normalized = []
    for s in spatters_list:
        if isinstance(s, np.ndarray):
            normalized.append(s)
        else:
            normalized.append(np.array(s, dtype=float) if len(s) > 0 else np.empty((0, 4)))
    spatters_list = normalized

    num_frames = len(spatters_list)

    cleaned = [None] * num_frames
    static = [None] * num_frames

    for i in range(num_frames):
        cleaned[i] = []
        static[i] = []

    # --- ПОИСК СТАТИЧНЫХ ---
    for i in range(num_frames - min_frames):
        frame0 = spatters_list[i]

        for j, sp0 in enumerate(frame0):
            cx0, cy0, w0, h0 = sp0
            is_static = True

            for t in range(1, min_frames):
                frame_t = spatters_list[i + t]
                found = False

                for sp in frame_t:
                    cx, cy, w, h = sp
                    if abs(cx - cx0) <= max_shift and abs(cy - cy0) <= max_shift:
                        found = True
                        break

                if not found:
                    is_static = False
                    break

            # --- КЛАССИФИКАЦИЯ ---
            if is_static:
                # добавляем в static на эти кадры
                for t in range(min_frames):
                    static[i + t].append(sp0.tolist())

            else:
                cleaned[i].append(sp0.tolist())

    # кадры после последнего окна просто копируем
    for i in range(num_frames - min_frames, num_frames):
        cleaned[i] = spatters_list[i].copy()

    # --- СТАВИМ ПУСТЫЕ np.array, где нужно ---
    cleaned = [np.array(f) if len(f) > 0 else np.empty((0, 4)) for f in cleaned]
    static = [np.array(f) if len(f) > 0 else np.empty((0, 4)) for f in static]

    return cleaned, static



def apply_dead_zone(spatters_list, welding_ellipses, dead_zone_radius=50):
    """
    Убирает брызги, попадающие в небольшую мертвую зону вокруг центра эллипса.
    """
    filtered = []

    for frame_idx, spatters in enumerate(spatters_list):
        if welding_ellipses[frame_idx] is None:
            # просто копируем
            filtered.append(spatters)
            continue

        # формат эллипса: ((xc, yc), (MA, ma), angle)
        (xc, yc), (MA, ma), angle = welding_ellipses[frame_idx]
        xc, yc = int(xc), int(yc)

        out = []

        for (cx, cy, w, h) in spatters:
            dist = np.hypot(cx - xc, cy - yc)

            # оставляем только те, которые ВНЕ dead zone
            if dist > dead_zone_radius:
                out.append([cx, cy, w, h])

        # приведение формата
        filtered.append(
            np.array(out) if len(out) > 0 else np.empty((0, 4))
        )

    return filtered



def apply_fixed_corridor_incremental(spatters_list, welding_ellipses, corridor_width=20, lookahead=5, show_progress=True):
    """
    Фильтрует брызги внутри "накопительного коридора".
    Коридор строится каждые lookahead кадров, накопительно.
    Фильтрация брызг на кадре idx учитывает только сегменты до idx.

    Параметры:
      spatters_list     — список массивов брызг по кадрам
      welding_ellipses  — список эллипсов зоны сварки по кадрам
      corridor_width    — ширина коридора
      lookahead         — количество кадров для расчета сегмента коридора
      show_progress     — показывать прогресс-бар в консоли
    """
    filtered = []
    corridor_segments = []
    first_center = None

    iterator = tqdm(enumerate(spatters_list), total=len(spatters_list), desc="Filtering spatters") if show_progress else enumerate(spatters_list)

    for idx, spatters in iterator:
        # --- Добавляем сегмент коридора, если есть эллипс на кадре ---
        ell = welding_ellipses[idx]
        if ell is not None:
            end_center = np.array(ell[0], dtype=float)
            if first_center is not None:
                segment = get_corridor_segment(first_center, end_center, corridor_width)
                corridor_segments.append(segment)
            first_center = end_center

        # --- Фильтруем брызги по накопленным сегментам ---
        out = []
        for (cx, cy, w, h) in spatters:
            point = np.array([cx, cy], dtype=float)
            inside_any = False
            for seg in corridor_segments:
                a, b, _, _ = seg
                seg_vec = np.array(b) - np.array(a)
                seg_length = np.linalg.norm(seg_vec)
                if seg_length < 1e-6:
                    continue
                seg_dir = seg_vec / seg_length
                seg_perp = np.array([-seg_dir[1], seg_dir[0]])
                rel = point - np.array(a)
                proj = np.dot(rel, seg_dir)
                side = np.dot(rel, seg_perp)
                if 0 <= proj <= seg_length and abs(side) <= corridor_width / 2:
                    inside_any = True
                    break
            if not inside_any:
                out.append([cx, cy, w, h])

        filtered.append(np.array(out) if out else np.empty((0, 4)))

    return filtered


def detect_hot_stripes_oriented(frame: np.ndarray,
                                threshold_low: int = 413,
                                threshold_high: int = 660,
                                min_area: int = 35,
                                circular_thr: float = 4.0,
                                angle_smooth: float = 0.3,
                                prev_angle: float = None):
    

    """
    Детекция вытянутых горячих полос с корректировкой угла.
    Фильтруются почти круглые эллипсы и слишком большие площади.
    Использует диапазон порогов (threshold_low, threshold_high).
    Возвращает список полос (x, y, w, h, angle) и обновленный угол шва.
    """
    if prev_angle is None:
        prev_angle = -3 * np.pi / 4

    # если верхний порог не задан, используем бесконечность
    if threshold_high is None:
        threshold_high = 65535 if frame.dtype != np.uint8 else 255

    # бинаризация по диапазону
    hot_mask = ((frame >= threshold_low) & (frame <= threshold_high)).astype(np.uint8)

    # морфология с вытянутым прямоугольным ядром
    kernel_len = 15
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 3))
    M = cv2.getRotationMatrix2D((kernel_len / 2, 1.5), degrees(prev_angle), 1.0)
    kernel_rot = cv2.warpAffine(ker, M, (kernel_len, 3))
    hot_dir = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel_rot, iterations=1)

    # поиск связных компонент
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(hot_dir)
    stripes = []
    angles_detected = []

    H, W = frame.shape[:2]
    max_allowed_area = 0.05 * (H * W)

    for i in range(1, num):
        mask_i = (labels == i).astype(np.uint8)
        area_pixels = cv2.countNonZero(mask_i)
        if area_pixels < min_area:
            continue

        pts = cv2.findNonZero(mask_i)
        if pts is not None and len(pts) >= 5:
            ellipse = cv2.fitEllipse(pts)
            (cx, cy), (MAJOR, MINOR), angle_deg = ellipse
            angle = radians(angle_deg)

            ellipse_area = pi * (MAJOR / 2) * (MINOR / 2)
            if ellipse_area > max_allowed_area:
                continue

            aspect = max(MAJOR, MINOR) / max(1e-5, min(MAJOR, MINOR))
            if aspect < circular_thr:
                continue
        else:
            x, y, w, h, _ = stats[i]
            cx, cy = x + w / 2, y + h / 2
            MAJOR, MINOR = max(w, h), min(w, h)
            angle = prev_angle

        stripes.append((cx, cy, MAJOR, MINOR, angle))
        angles_detected.append(angle)

    # обновление глобального угла
    if angles_detected:
        new_angle = np.mean(angles_detected)
        updated_angle = prev_angle + angle_smooth * (new_angle - prev_angle)
    else:
        updated_angle = prev_angle

    return stripes, updated_angle



def apply_hot_stripes_filter_oriented(cleaned_spatters, stripes):
    """
    Убирает капли, попавшие в вытянутые горячие полосы.
    cleaned_spatters: list of [x, y, w, h]
    stripes: list of (x, y, w, h, angle)
    """
    filtered = []
    for sp in cleaned_spatters:
        sx, sy = sp[0], sp[1]
        keep = True
        for x, y, w, h, angle in stripes:
            cx, cy = x + w / 2, y + h / 2
            long_side, short_side = max(w, h), min(w, h)
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            dx, dy = sx - cx, sy - cy
            dx_r = cos_a * dx - sin_a * dy
            dy_r = sin_a * dx + cos_a * dy
            if abs(dx_r) <= long_side / 2 and abs(dy_r) <= short_side / 2:
                keep = False
                break
        if keep:
            filtered.append(sp)
    return filtered

def filter_spatters_by_hot_stripes_video_oriented(frames: np.ndarray,
                                                  cleaned_spatters_list,
                                                  threshold_low=413,
                                                  threshold_high=660,   # можно задать верхний порог
                                                  min_area=35,
                                                  circular_thr=4.0,
                                                  angle_smooth=0.3,
                                                  initial_angle=-3*np.pi/4,
                                                  show_progress=True):
    """
    Покадровая фильтрация всех капель с учетом ориентации полос.
    """
    cleaned_spatters_new = []
    hot_stripes_list = []
    current_angle = initial_angle

    iterator = range(len(frames))
    if show_progress:
        iterator = tqdm(iterator, desc="Filtering spatters by oriented hot stripes")

    for i in iterator:
        frame = frames[i]
        cleaned_spatters = cleaned_spatters_list[i]

        stripes, current_angle = detect_hot_stripes_oriented(
            frame=frame,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_area=min_area,
            circular_thr=circular_thr,
            angle_smooth=angle_smooth,
            prev_angle=current_angle
        )

        cleaned = apply_hot_stripes_filter_oriented(cleaned_spatters, stripes)
        cleaned_spatters_new.append(cleaned)
        hot_stripes_list.append(stripes)

    return cleaned_spatters_new, hot_stripes_list


def detect_and_filter_spatters_by_hot_stripes(frames: np.ndarray,
                                              cleaned_spatters_list,
                                              hot_threshold=180,
                                              min_area=20,
                                              circular_thr=4.0,
                                              angle_smooth=0.3,
                                              initial_angle=radians(45),
                                              show_progress=True):
    """
    Полная покадровая обработка:
    1) Детекция вытянутых горячих полос с учетом ориентации.
    2) Фильтрация капель внутри этих полос.
    """
    hot_stripes_list = []
    cleaned_spatters_new = []
    current_angle = initial_angle

    iterator = range(len(frames))
    if show_progress:
        iterator = tqdm(iterator, desc="Detecting and filtering hot stripes")

    for i in iterator:
        frame = frames[i]
        cleaned_spatters = cleaned_spatters_list[i]

        stripes, current_angle = detect_hot_stripes_oriented(
            frame=frame,
            hot_threshold=hot_threshold,
            min_area=min_area,
            circular_thr=circular_thr,
            angle_smooth=angle_smooth,
            prev_angle=current_angle
        )

        hot_stripes_list.append(stripes)
        cleaned = apply_hot_stripes_filter_oriented(cleaned_spatters, stripes)
        cleaned_spatters_new.append(cleaned)

    return cleaned_spatters_new, hot_stripes_list