import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict

class SpatterTracker:
    def __init__(self, max_mahalanobis_distance=10, max_frames_skipped=3, 
                 process_noise=1.0, measurement_noise=1.0, 
                 frame_margin=10, max_velocity=100,
                 min_movement_distance=5, min_movement_frames=3):
        """
        Инициализация трекера брызг
        
        Args:
            max_mahalanobis_distance: максимальное расстояние Махаланобиса для сопоставления
            max_frames_skipped: максимальное число пропущенных кадров
            process_noise: шум процесса для фильтра Калмана
            measurement_noise: шум измерений для фильтра Калмана
            frame_margin: запас за границами кадра для удаления треков
            max_velocity: максимальная физически возможная скорость (пикселей/кадр)
            min_movement_distance: минимальное расстояние перемещения за период (пикселей)
            min_movement_frames: минимальное количество кадров для анализа движения
        """
        self.max_mahalanobis_distance = max_mahalanobis_distance
        self.max_frames_skipped = max_frames_skipped
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.frame_margin = frame_margin
        self.max_velocity = max_velocity
        self.min_movement_distance = min_movement_distance
        self.min_movement_frames = min_movement_frames
        
        self.tracks = {}  # {id: {'filter': KalmanFilter, 'frames_skipped': int, 'positions': list, 'is_static': bool}}
        self.next_id = 0
        self.frame_size = None
        
    def _create_kalman_filter(self, x, y, w, h):
        """Создает фильтр Калмана для трека"""
        kf = KalmanFilter(dim_x=6, dim_z=4)
        
        # Матрица состояния: [x, y, vx, vy, w, h]
        kf.F = np.array([[1, 0, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        
        # Матрица измерений: измеряем только позицию и размер
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        
        # Ковариация процесса
        kf.Q = np.eye(6) * self.process_noise
        
        # Ковариация измерений
        kf.R = np.eye(4) * self.measurement_noise
        
        # Начальное состояние
        kf.x = np.array([x, y, 0, 0, w, h])
        kf.P = np.eye(6) * 100  # Большая начальная неопределенность
        
        return kf
    
    def _get_measurements(self, spatters):
        """Извлекает измерения из детекций"""
        measurements = []
        for spatter in spatters:
            x, y, w, h = spatter
            measurements.append(np.array([x, y, w, h]))
        return measurements
    
    def _association(self, predictions, measurements):
        """Сопоставляет предсказания с измерениями используя венгерский алгоритм"""
        if not predictions or not measurements:
            return [], list(range(len(predictions))), list(range(len(measurements)))
        
        cost_matrix = np.zeros((len(predictions), len(measurements)))
        
        for i, (track_id, pred) in enumerate(predictions):
            for j, meas in enumerate(measurements):
                # Используем евклидово расстояние для простоты
                distance = np.linalg.norm(pred[:2] - meas[:2])
                cost_matrix[i, j] = distance
        
        # Венгерский алгоритм для минимизации общей стоимости
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_predictions = list(range(len(predictions)))
        unmatched_measurements = list(range(len(measurements)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_mahalanobis_distance:
                matches.append((i, j))
                unmatched_predictions.remove(i)
                unmatched_measurements.remove(j)
        
        return matches, unmatched_predictions, unmatched_measurements
    
    def _is_static_spatter(self, track_id: int) -> bool:
        """
        Проверяет, является ли брызга статичной (колеблется около одной точки)
        
        Returns:
            True если брызга движется меньше min_movement_distance за min_movement_frames
        """
        if track_id not in self.tracks:
            return False
            
        track = self.tracks[track_id]
        if 'positions' not in track or len(track['positions']) < self.min_movement_frames:
            return False
        
        positions = track['positions']
        
        # Берем последние N позиций
        recent_positions = positions[-self.min_movement_frames:]
        
        # Вычисляем общее пройденное расстояние
        total_distance = 0
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            total_distance += distance
        
        # Если общее расстояние меньше порога - считаем статичной
        is_static = total_distance < self.min_movement_distance
        
        # Помечаем трек как статичный
        if is_static:
            self.tracks[track_id]['is_static'] = True
        
        return is_static
    
    def update(self, spatters: List[np.ndarray], frame_width: int, frame_height: int) -> Tuple[List, int, int]:
        """
        Обновляет треки на основе новых детекций
        
        Returns:
            Tuple: (spatters_with_ids, new_tracks_count, lost_tracks_count)
            spatters_with_ids: список [x, y, w, h, id] только для НЕстатичных брызг
        """
        if self.frame_size is None:
            self.frame_size = (frame_width, frame_height)
        
        # Шаг предсказания для всех активных треков
        predictions = []
        track_ids = list(self.tracks.keys())
        
        for track_id in track_ids:
            track = self.tracks[track_id]
            track['filter'].predict()
            pred_state = track['filter'].x
            
            # Проверяем не вышел ли трек за границы
            x, y = pred_state[0], pred_state[1]
            if (x < -self.frame_margin or x > frame_width + self.frame_margin or 
                y < -self.frame_margin or y > frame_height + self.frame_margin):
                del self.tracks[track_id]
                continue
            
            # Проверяем физическую возможность скорости
            velocity = np.linalg.norm(pred_state[2:4])
            if velocity > self.max_velocity:
                del self.tracks[track_id]
                continue
            
            # Проверяем на статичность (только если накопилось достаточно кадров)
            if len(track.get('positions', [])) >= self.min_movement_frames:
                if self._is_static_spatter(track_id):
                    # Помечаем как статичный, но не удаляем сразу (для статистики)
                    continue
            
            predictions.append((track_id, pred_state))
        
        # Получаем измерения
        measurements = self._get_measurements(spatters)
        
        # Сопоставление
        matches, unmatched_predictions, unmatched_measurements = self._association(
            predictions, measurements)
        
        # Создаем список брызг с ID - ТОЛЬКО НЕСТАТИЧНЫЕ
        spatters_with_ids = []
        
        # Обновление сопоставленных треков
        for pred_idx, meas_idx in matches:
            track_id = predictions[pred_idx][0]
            measurement = measurements[meas_idx]
            self.tracks[track_id]['filter'].update(measurement)
            self.tracks[track_id]['frames_skipped'] = 0
            
            # Сохраняем историю позиций
            if 'positions' not in self.tracks[track_id]:
                self.tracks[track_id]['positions'] = []
            x, y, w, h = measurement
            self.tracks[track_id]['positions'].append((x, y))
            
            # Ограничиваем длину истории
            if len(self.tracks[track_id]['positions']) > self.min_movement_frames * 2:
                self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-self.min_movement_frames * 2:]
            
            # Добавляем брызг с ID ТОЛЬКО если он не статичный
            if not self.tracks[track_id].get('is_static', False):
                spatters_with_ids.append([x, y, w, h, track_id])
        
        # Обновление несопоставленных треков (пропуск кадра)
        lost_tracks_count = 0
        for pred_idx in unmatched_predictions:
            track_id = predictions[pred_idx][0]
            self.tracks[track_id]['frames_skipped'] += 1
            if self.tracks[track_id]['frames_skipped'] > self.max_frames_skipped:
                del self.tracks[track_id]
                lost_tracks_count += 1
        
        # Создание новых треков для несопоставленных измерений
        new_tracks_count = 0
        for meas_idx in unmatched_measurements:
            measurement = measurements[meas_idx]
            x, y, w, h = measurement
            kf = self._create_kalman_filter(x, y, w, h)
            self.tracks[self.next_id] = {
                'filter': kf,
                'frames_skipped': 0,
                'positions': [(x, y)],  # Начинаем собирать историю позиций
                'is_static': False  # Новые треки по умолчанию не статичные
            }
            
            # Добавляем новый брызг с ID
            spatters_with_ids.append([x, y, w, h, self.next_id])
            self.next_id += 1
            new_tracks_count += 1
        
        return spatters_with_ids, new_tracks_count, lost_tracks_count

def track_spatters_with_ids(cleaned_spatters: List[np.ndarray], 
                          frame_width: int, 
                          frame_height: int,
                          max_mahalanobis_distance: float = 10,
                          max_frames_skipped: int = 3,
                          process_noise: float = 1.0,
                          measurement_noise: float = 1.0,
                          frame_margin: int = 10,
                          max_velocity: float = 100,
                          min_movement_distance: float = 5,
                          min_movement_frames: int = 3) -> Tuple[List, pd.DataFrame]:
    """
    Трекинг брызг с назначением ID и расчет статистики
    
    Args:
        cleaned_spatters: Список массивов с детекциями брызг по кадрам
        frame_width: Ширина кадра
        frame_height: Высота кадра
        max_mahalanobis_distance: Макс. расстояние для сопоставления треков
        max_frames_skipped: Макс. число пропущенных кадров перед удалением трека
        process_noise: Шум процесса для фильтра Калмана
        measurement_noise: Шум измерений для фильтра Калмана
        frame_margin: Запас за границами кадра для удаления треков
        max_velocity: Макс. физически возможная скорость (пикселей/кадр)
        min_movement_distance: Мин. расстояние перемещения для фильтрации статичных брызг
        min_movement_frames: Мин. количество кадров для анализа движения
    
    Returns:
        Tuple: (cleaned_spatters_id, stats_dataframe)
        - cleaned_spatters_id: список массивов [x, y, w, h, id] только для НЕстатичных брызг
        - stats_dataframe: DataFrame со статистикой по кадрам
    """
    
    tracker = SpatterTracker(
        max_mahalanobis_distance=max_mahalanobis_distance,
        max_frames_skipped=max_frames_skipped,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        frame_margin=frame_margin,
        max_velocity=max_velocity,
        min_movement_distance=min_movement_distance,
        min_movement_frames=min_movement_frames
    )
    
    cleaned_spatters_id = []
    results = []
    
    for frame_idx, frame_spatters in enumerate(cleaned_spatters):
        # Обновляем трекер и получаем брызги с ID
        spatters_with_ids, new_spatters, lost_spatters = tracker.update(
            frame_spatters, frame_width, frame_height)
        
        cleaned_spatters_id.append(np.array(spatters_with_ids) if spatters_with_ids else np.array([]))
        
        # Рассчитываем статистику для активных треков - ТОЛЬКО НЕСТАТИЧНЫХ
        velocities = []
        areas = []
        static_count = 0
        
        for track_id, track_data in tracker.tracks.items():
            # Пропускаем статичные треки в статистике скорости и площади
            if track_data.get('is_static', False):
                static_count += 1
                continue
                
            # Скорость из состояния фильтра Калмана
            velocity = np.linalg.norm(track_data['filter'].x[2:4])
            velocities.append(velocity)
            
            # Площадь брызги
            w, h = track_data['filter'].x[4], track_data['filter'].x[5]
            area = w * h
            areas.append(area)
        
        # Собираем статистику по кадру
        frame_stats = {
            'frame_number': frame_idx + 1,
            'spatters_total': len(tracker.tracks) - static_count,  # только нестатические
            'spatters_new': new_spatters,
            'spatters_lost': lost_spatters,
            'spatters_mean_velocity': np.mean(velocities) if velocities else 0,
            'spatters_mean_area': np.mean(areas) if areas else 0
        }
        
        results.append(frame_stats)
    
    return cleaned_spatters_id, pd.DataFrame(results)