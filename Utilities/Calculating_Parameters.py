import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_spatters_count_dataframe(cleaned_spatters, thermogram_id):
    """
    Создает датафрейм с количеством брызг на каждый кадр
    
    Args:
        cleaned_spatters: список массивов брызг по кадрам
        thermogram_id: номер термограммы (опционально)
    
    Returns:
        DataFrame с колонками ['frame_number', 'spatters_count']
    """
    
    # Создаем список с количеством брызг на каждый кадр
    spatters_count = [len(frame_spatters) for frame_spatters in cleaned_spatters]
    
    # Создаем датафрейм
    df = pd.DataFrame({
        'frame_number': range(1, len(cleaned_spatters) + 1),
        'spatters_count_del': spatters_count #возьмем из трекера
    })
    
    # Добавляем ID термограммы если указан
    if thermogram_id is not None:
        df['thermogram_id'] = thermogram_id
    
    return df


def calculate_spatters_median_temperature(cleaned_spatters, frames, threshold_temp=420):
    """
    Рассчитывает медианную температуру брызг для каждого кадра с применением порога температуры.

    Аргументы:
        cleaned_spatters: список массивов брызг по кадрам в формате [cx, cy, w, h]
        frames: исходные кадры с температурными данными (numpy arrays)
        threshold_temp: порог температуры - пиксели ниже этого значения игнорируются при расчетах
        thermogram_id: идентификатор термограммы для группировки результатов

    Возвращает:
        DataFrame со следующими колонками:
        - 'frame_number': номер кадра (начинается с 1)
        - 'spatters_count': общее количество обнаруженных брызг на кадре
        - 'spatters_above_threshold': количество брызг, имеющих хотя бы один пиксель выше порога
        - 'median_spatter_temperature': медианная температура брызг (рассчитанная только по пикселям выше порога)
        - 'mean_spatter_temperature': средняя температура брызг (рассчитанная только по пикселям выше порога)
        - 'max_spatter_temperature': максимальная температура среди всех брызг (по пикселям выше порога)
        - 'total_pixels_above_threshold': общее количество пикселей во всех брызгах, превышающих порог
        - 'percentage_above_threshold': процент пикселей выше порога от общего количества пикселей в брызгах
        - 'threshold_used': значение использованного порога температуры
        - 'thermogram_id': идентификатор термограммы (если передан)

    Примечания:
        - Для кадров без брызг все температурные показатели устанавливаются в np.nan
        - Расчет температур производится только для пикселей, превышающих заданный порог
        - Каждая брызга учитывается в статистике только если содержит хотя бы один пиксель выше порога
    """
    results = []
    
    for frame_idx in range(len(cleaned_spatters)):
        frame_spatters = cleaned_spatters[frame_idx]
        frame_data = frames[frame_idx]
        
        if len(frame_spatters) == 0:
            # Если брызг нет, записываем NaN
            frame_result = {
                'frame_number': frame_idx + 1,
                'spatters_above_threshold': 0,
                'spatters_median_temperature': np.nan,
                'spatters_mean_temperature': np.nan,
                'spatters_max_temperature': np.nan,
                'spatters_total_pixels_above_threshold': 0,
                'spatters_percentage_above_threshold': 0.0
            }
        else:
            spatter_mean_temperatures = []
            spatters_above_threshold = 0
            total_pixels_above = 0
            total_pixels_all = 0
            
            for spatter in frame_spatters:
                cx, cy, w, h = spatter.astype(int)
                
                # Вычисляем bounding box брызги
                x_start = max(0, cx - w//2)
                x_end = min(frame_data.shape[1], cx + w//2 + 1)
                y_start = max(0, cy - h//2)
                y_end = min(frame_data.shape[0], cy + h//2 + 1)
                
                # Извлекаем пиксели брызги
                spatter_pixels = frame_data[y_start:y_end, x_start:x_end]
                
                if spatter_pixels.size > 0:
                    # ПРИМЕНЯЕМ ПОРОГ: оставляем только пиксели выше threshold_temp
                    above_threshold_pixels = spatter_pixels[spatter_pixels >= threshold_temp]
                    total_pixels_all += spatter_pixels.size
                    total_pixels_above += above_threshold_pixels.size
                    
                    if above_threshold_pixels.size > 0:
                        # Средняя температура для одной брызги (только выше порога)
                        mean_temp = np.mean(above_threshold_pixels)
                        spatter_mean_temperatures.append(mean_temp)
                        spatters_above_threshold += 1
            
            # Рассчитываем статистики
            if spatter_mean_temperatures:
                median_temp = np.median(spatter_mean_temperatures)
                mean_temp = np.mean(spatter_mean_temperatures)
                max_temp = np.max(spatter_mean_temperatures)
                percentage_above = (total_pixels_above / total_pixels_all * 100) if total_pixels_all > 0 else 0
            else:
                median_temp = np.nan
                mean_temp = np.nan
                max_temp = np.nan
                percentage_above = 0.0
            
            frame_result = {
                'frame_number': frame_idx + 1,
                'spatters_above_threshold': spatters_above_threshold,
                'spatters_median_temperature': median_temp,
                'spatters_mean_temperature': mean_temp,
                'spatters_max_temperature': max_temp,
                'spatters_total_pixels_above_threshold': total_pixels_above,
                'spatters_percentage_above_threshold': percentage_above,
            }
        
        results.append(frame_result)
    
    df = pd.DataFrame(results)

    return df


def visualize_spatters_threshold(frame_idx, cleaned_spatters, frames, threshold_temp=420, max_spatters_to_show=10):
    """
    Визуализирует брызги с применением порога температуры для настройки
    
    Аргументы:
        frame_idx: номер кадра для визуализации
        cleaned_spatters: список массивов брызг
        frames: кадры с температурными данными
        threshold_temp: порог температуры
        max_spatters_to_show: максимальное количество брызг для отображения
    """
    
    frame_spatters = cleaned_spatters[frame_idx]
    frame_data = frames[frame_idx]
    
    if len(frame_spatters) == 0:
        print(f"На кадре {frame_idx} брызг не обнаружено")
        return
    
    print(f"=== АНАЛИЗ БРЫЗГ НА КАДРЕ {frame_idx} ===")
    print(f"Всего брызг: {len(frame_spatters)}")
    print(f"Порог температуры: {threshold_temp}")
    print("-" * 50)
    
    # Статистика по всем брызгам
    all_spatter_stats = []
    
    for i, spatter in enumerate(frame_spatters[:max_spatters_to_show]):
        cx, cy, w, h = spatter.astype(int)
        
        # Вычисляем bounding box
        x_start = max(0, cx - w//2)
        x_end = min(frame_data.shape[1], cx + w//2 + 1)
        y_start = max(0, cy - h//2)
        y_end = min(frame_data.shape[0], cy + h//2 + 1)
        
        # Извлекаем пиксели
        spatter_pixels = frame_data[y_start:y_end, x_start:x_end]
        
        if spatter_pixels.size > 0:
            # Пиксели выше и ниже порога
            above_threshold = spatter_pixels[spatter_pixels >= threshold_temp]
            below_threshold = spatter_pixels[spatter_pixels < threshold_temp]
            
            spatter_stat = {
                'spatter_id': i + 1,
                'center': (cx, cy),
                'size': (w, h),
                'total_pixels': spatter_pixels.size,
                'pixels_above_threshold': above_threshold.size,
                'pixels_below_threshold': below_threshold.size,
                'percentage_above': (above_threshold.size / spatter_pixels.size * 100) if spatter_pixels.size > 0 else 0,
                'mean_temp_above': np.mean(above_threshold) if above_threshold.size > 0 else np.nan,
                'max_temp_above': np.max(above_threshold) if above_threshold.size > 0 else np.nan,
                'mean_temp_all': np.mean(spatter_pixels),
                'max_temp_all': np.max(spatter_pixels)
            }
            
            all_spatter_stats.append(spatter_stat)
            
            print(f"Брызг {i+1}:")
            print(f"  Центр: ({cx}, {cy}), Размер: {w}x{h}")
            print(f"  Пикселей: {spatter_pixels.size}")
            print(f"  Выше порога: {above_threshold.size} ({spatter_stat['percentage_above']:.1f}%)")
            if above_threshold.size > 0:
                print(f"  Средняя темп. выше порога: {spatter_stat['mean_temp_above']:.2f}")
                print(f"  Макс. темп. выше порога: {spatter_stat['max_temp_above']:.2f}")
            print(f"  Средняя темп. всех пикселей: {spatter_stat['mean_temp_all']:.2f}")
            print(f"  Макс. темп. всех пикселей: {spatter_stat['max_temp_all']:.2f}")
            print()
    
    # Общая статистика
    if all_spatter_stats:
        total_spatters_above = sum(1 for s in all_spatter_stats if s['pixels_above_threshold'] > 0)
        avg_percentage_above = np.mean([s['percentage_above'] for s in all_spatter_stats])
        
        print("ОБЩАЯ СТАТИСТИКА:")
        print(f"Брызг выше порога: {total_spatters_above}/{len(all_spatter_stats)}")
        print(f"Средний % пикселей выше порога: {avg_percentage_above:.1f}%")
    
    # ВИЗУАЛИЗАЦИЯ
    fig, axes = plt.subplots(1, 3, figsize=(15, 12))
    
    # 1. Исходный кадр с bounding boxes
    axes[0].imshow(frame_data, cmap='hot')
    for i, spatter in enumerate(frame_spatters[:max_spatters_to_show]):
        cx, cy, w, h = spatter.astype(int)
        rect = plt.Rectangle((cx - w//2, cy - h//2), w, h, 
                           fill=False, color='red', linewidth=1.5)
        axes[0].add_patch(rect)
        axes[0].text(cx, cy - h//2 - 5, f'{i+1}', color='white', 
                       fontsize=8, ha='center', va='bottom')
    axes[0].set_title(f'Кадр {frame_idx} - Bounding Boxes брызг')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # 2. Кадр с применением порога (пиксели ниже порога - черные)
    display_frame = frame_data.copy()
    for spatter in frame_spatters:
        cx, cy, w, h = spatter.astype(int)
        x_start = max(0, cx - w//2)
        x_end = min(frame_data.shape[1], cx + w//2 + 1)
        y_start = max(0, cy - h//2)
        y_end = min(frame_data.shape[0], cy + h//2 + 1)
        
        # Обнуляем пиксели ниже порога в bounding boxes
        spatter_region = display_frame[y_start:y_end, x_start:x_end]
        below_threshold_mask = spatter_region < threshold_temp
        spatter_region[below_threshold_mask] = 0
    
    axes[1].imshow(display_frame, cmap='hot')
    axes[1].set_title(f'Кадр {frame_idx} - Пиксели выше {threshold_temp}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # 3. Гистограмма температур всех пикселей брызг
    all_spatter_pixels = []
    for spatter in frame_spatters:
        cx, cy, w, h = spatter.astype(int)
        x_start = max(0, cx - w//2)
        x_end = min(frame_data.shape[1], cx + w//2 + 1)
        y_start = max(0, cy - h//2)
        y_end = min(frame_data.shape[0], cy + h//2 + 1)
        
        spatter_pixels = frame_data[y_start:y_end, x_start:x_end]
        all_spatter_pixels.extend(spatter_pixels.flatten())
    
    if all_spatter_pixels:
        axes[2].hist(all_spatter_pixels, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[2].axvline(x=threshold_temp, color='red', linestyle='--', 
                          linewidth=2, label=f'Порог: {threshold_temp}')
        axes[2].set_xlabel('Температура')
        axes[2].set_ylabel('Количество пикселей')
        axes[2].set_title('Распределение температур всех пикселей брызг')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_spatter_stats


def calculate_welding_zone_temperature_threshold(welding_ellipses, frames, threshold_temp=400, radius=50, thermogram_id=None):
    """
    Рассчитывает среднюю температуру в зоне сварки с применением порога температуры
    
    Аргументы:
        welding_ellipses: список эллипсов зоны сварки по кадрам
        frames: исходные кадры с температурными данными
        threshold_temp: порог температуры (пиксели ниже этого значения игнорируются)
        radius: радиус зоны вокруг центра эллипса
        thermogram_id: номер термограммы
    
    Возвращает:
        DataFrame со средней и максимальной температурой зоны сварки по кадрам после применения порога
        и размер зона сварки выше порога в пикселях
    """
    
    results = []
    
    for frame_idx in range(len(welding_ellipses)):
        frame_ellipse = welding_ellipses[frame_idx]
        frame_data = frames[frame_idx]
        
        if frame_ellipse is None:
            # Если зоны сварки нет, записываем NaN
            frame_result = {
                'frame_number': frame_idx + 1,
                'welding_zone_detected': False,
                'welding_zone_mean_temperature': np.nan,
                'welding_zone_max_temperature': np.nan,
                'welding_zone_size': 0
            }
        else:
            # Извлекаем центр эллипса
            center_x, center_y = frame_ellipse[0]
            center_x, center_y = int(center_x), int(center_y)
            
            # Определяем границы круговой зоны
            x_start = max(0, center_x - radius)
            x_end = min(frame_data.shape[1], center_x + radius + 1)
            y_start = max(0, center_y - radius)
            y_end = min(frame_data.shape[0], center_y + radius + 1)
            
            # Извлекаем пиксели в круговой зоне
            welding_zone_pixels = frame_data[y_start:y_end, x_start:x_end]
            
            # Создаем маску для круговой области
            y_coords, x_coords = np.ogrid[y_start:y_end, x_start:x_end]
            distance_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
            
            # Применяем круговую маску и порог температуры
            masked_pixels = welding_zone_pixels[distance_mask]
            above_threshold_pixels = masked_pixels[masked_pixels >= threshold_temp]
            
            if above_threshold_pixels.size > 0:
                # Средняя температура пикселей выше порога
                mean_temp = np.mean(above_threshold_pixels)
                max_temp = np.max(above_threshold_pixels)
                pixels_count = above_threshold_pixels.size
            else:
                mean_temp = np.nan
                max_temp = np.nan
                pixels_count = 0
            
            frame_result = {
                'frame_number': frame_idx + 1,
                'welding_zone_detected': True,
                'welding_zone_mean_temperature': mean_temp,
                'welding_zone_max_temperature': max_temp,
                'welding_zone_size': pixels_count
            }
        
        results.append(frame_result)
    
    df = pd.DataFrame(results)
    
    # Добавляем колонку test_zone
    df['test_zone'] = None
    
    # Находим кадры где welding_zone_size > 10
    valid_frames = df[df['welding_zone_size'] > 10]
    
    if len(valid_frames) > 0:
        # Находим первый и последний кадры с welding_zone_size > 10
        first_frame = valid_frames['frame_number'].min()
        last_frame = valid_frames['frame_number'].max()
        
        # Вычисляем общее количество кадров в интервале
        total_frames_in_interval = last_frame - first_frame + 1
        
        # Делим на 4 равные части
        quarter_size = total_frames_in_interval // 4
        
        # Назначаем метки A, B, C, D для каждой четверти
        for i in range(4):
            start_frame = first_frame + i * quarter_size
            end_frame = first_frame + (i + 1) * quarter_size - 1 if i < 3 else last_frame
            
            # Определяем метку для текущей четверти
            zone_label = ['A', 'B', 'C', 'D'][i]
            
            # Записываем метку в test_zone для кадров этой четверти
            mask = (df['frame_number'] >= start_frame) & (df['frame_number'] <= end_frame)
            df.loc[mask, 'test_zone'] = zone_label
    
    return df


def visualize_welding_zone_threshold(frame_idx, welding_ellipses, frames, threshold_temp=400, radius=50):
    """
    Визуализирует зону сварки с применением порога температуры для настройки
    
    Аргументы:
        frame_idx: номер кадра для визуализации
        welding_ellipses: список эллипсов зоны сварки
        frames: кадры с температурными данными
        threshold_temp: порог температуры
        radius: радиус зоны
    """
    
    frame_ellipse = welding_ellipses[frame_idx]
    frame_data = frames[frame_idx]
    
    if frame_ellipse is None:
        print(f"На кадре {frame_idx} зона сварки не обнаружена")
        return
    
    # Извлекаем центр эллипса
    center_x, center_y = frame_ellipse[0]
    center_x, center_y = int(center_x), int(center_y)
    
    # Определяем границы круговой зоны
    x_start = max(0, center_x - radius)
    x_end = min(frame_data.shape[1], center_x + radius + 1)
    y_start = max(0, center_y - radius)
    y_end = min(frame_data.shape[0], center_y + radius + 1)
    
    # Извлекаем пиксели в круговой зоне
    welding_zone_pixels = frame_data[y_start:y_end, x_start:x_end]
    
    # Создаем маску для круговой области
    y_coords, x_coords = np.ogrid[y_start:y_end, x_start:x_end]
    distance_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
    
    # Применяем круговую маску и порог температуры
    masked_pixels = welding_zone_pixels[distance_mask]
    above_threshold_pixels = masked_pixels[masked_pixels >= threshold_temp]
    below_threshold_pixels = masked_pixels[masked_pixels < threshold_temp]
    
    # Статистика
    total_pixels = masked_pixels.size
    above_count = above_threshold_pixels.size
    below_count = below_threshold_pixels.size
    
    print(f"=== АНАЛИЗ КАДРА {frame_idx} ===")
    print(f"Центр зоны: ({center_x}, {center_y})")
    print(f"Радиус зоны: {radius} пикселей")
    print(f"Порог температуры: {threshold_temp}")
    print(f"Всего пикселей в зоне: {total_pixels}")
    print(f"Пикселей выше порога: {above_count} ({above_count/total_pixels*100:.1f}%)")
    print(f"Пикселей ниже порога: {below_count} ({below_count/total_pixels*100:.1f}%)")
    
    if above_count > 0:
        print(f"Температура выше порога:")
        print(f"  Средняя: {np.mean(above_threshold_pixels):.2f}")
        print(f"  Максимальная: {np.max(above_threshold_pixels):.2f}")
        print(f"  Минимальная: {np.min(above_threshold_pixels):.2f}")
        print(f"  Медианная: {np.median(above_threshold_pixels):.2f}")
    
    print(f"Общая статистика зоны:")
    print(f"  Средняя температура всех пикселей: {np.mean(masked_pixels):.2f}")
    print(f"  Максимальная температура: {np.max(masked_pixels):.2f}")
    print(f"  Минимальная температура: {np.min(masked_pixels):.2f}")
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Исходный кадр с зоной (все что ниже порога - черным)
    display_frame = frame_data.copy()
    
    # Создаем маску для всей зоны сварки
    y_indices, x_indices = np.indices(frame_data.shape)
    zone_mask = ((x_indices - center_x)**2 + (y_indices - center_y)**2) <= radius**2
    
    # Все что ниже порога в зоне сварки делаем черным
    below_threshold_mask = zone_mask & (frame_data < threshold_temp)
    display_frame[below_threshold_mask] = 0  # черный цвет
    
    axes[0].imshow(display_frame, cmap='hot')
    axes[0].add_patch(plt.Circle((center_x, center_y), radius, fill=False, color='red', linewidth=2))
    axes[0].plot(center_x, center_y, 'rx', markersize=10)
    axes[0].set_title(f'Кадр {frame_idx} - Зона сварки\n(ниже {threshold_temp} - черный)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # 2. Гистограмма температур в зоне
    axes[1].hist(masked_pixels, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(x=threshold_temp, color='red', linestyle='--', linewidth=2, label=f'Порог: {threshold_temp}')
    axes[1].set_xlabel('Температура')
    axes[1].set_ylabel('Количество пикселей')
    axes[1].set_title('Распределение температур в зоне')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Сравнение пикселей выше/ниже порога
    temperatures_above = above_threshold_pixels if above_count > 0 else np.array([])
    temperatures_below = below_threshold_pixels if below_count > 0 else np.array([])
    
    if above_count > 0 or below_count > 0:
        data_to_plot = []
        labels = []
        if below_count > 0:
            data_to_plot.append(temperatures_below)
            labels.append(f'Ниже порога\n({below_count} пикс.)')
        if above_count > 0:
            data_to_plot.append(temperatures_above)
            labels.append(f'Выше порога\n({above_count} пикс.)')
        
        axes[2].boxplot(data_to_plot, labels=labels)
        axes[2].set_ylabel('Температура')
        axes[2].set_title('Сравнение температур по порогу')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return above_threshold_pixels
