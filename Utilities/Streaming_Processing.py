from Utilities.Detect import detect_spatters_video, detect_welding_zone_video, apply_dead_zone, apply_fixed_corridor_incremental, filter_spatters_by_hot_stripes_video_oriented, remove_static_spatters
from Utilities.Calculating_Parameters import create_spatters_count_dataframe, calculate_spatters_median_temperature, calculate_welding_zone_temperature_threshold
from Utilities.Spatter_Tracker import track_spatters_with_ids
import pandas as  pd
import numpy as np
import os
import pickle


def process_multiple_thermograms_safe(thermogram_ids, 
                                    original_base_path='../Data/NumPy_convert',
                                    filtered_base_path='../Data/Filtered',
                                    save_dir='../Debugging_Information/Progress_Streaming_Processing'):
    """
    Безопасная обработка нескольких термограмм с сохранением прогресса
    
    Args:
        thermogram_ids: список номеров термограмм для обработки
        original_base_path: путь к оригинальным файлам (thermogram_X.npy)
        filtered_base_path: путь к фильтрованным файлам (thermogram_X_filtered.npy)
        save_dir: директория для сохранения прогресса
    
    Returns:
        DataFrame с объединенными данными по всем обработанным термограммам
    """
    
    # Создаем директорию для сохранения прогресса, если ее нет
    os.makedirs(save_dir, exist_ok=True)
    
    # Пути к файлам прогресса
    df_save_path = f'{save_dir}/df_combined_all.pkl'
    ids_save_path = f'{save_dir}/processed_thermogram_ids.pkl'
    
    # 1. ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ПРОГРЕССА
    processed_thermogram_ids = []
    df_combined_all = pd.DataFrame()
    
    # Загружаем список уже обработанных ID
    if os.path.exists(ids_save_path):
        try:
            with open(ids_save_path, 'rb') as f:
                processed_thermogram_ids = pickle.load(f)
            print(f"✓ Загружены обработанные термограммы: {len(processed_thermogram_ids)} шт")
        except Exception as e:
            print(f"✗ Ошибка загрузки списка ID: {e}")
    
    # Загружаем сохраненные данные
    if os.path.exists(df_save_path):
        try:
            df_combined_all = pd.read_pickle(df_save_path)
            print(f"✓ Загружены сохраненные данные: {len(df_combined_all)} строк")
            if 'thermogram_id' in df_combined_all.columns:
                print(f"✓ В данных присутствуют термограммы: {sorted(df_combined_all['thermogram_id'].unique())}")
        except Exception as e:
            print(f"✗ Ошибка загрузки данных: {e}")
    
    # 2. ОПРЕДЕЛЕНИЕ НЕОБРАБОТАННЫХ ТЕРМОГРАММ
    unprocessed_ids = [tid for tid in thermogram_ids if tid not in processed_thermogram_ids]
    
    print(f"\n{'='*50}")
    print("СТАТУС ОБРАБОТКИ")
    print(f"{'='*50}")
    print(f"Всего запрошено: {len(thermogram_ids)} термограмм")
    print(f"Уже обработано: {len(processed_thermogram_ids)}")
    print(f"Нужно обработать: {len(unprocessed_ids)}")
    
    if len(processed_thermogram_ids) > 0:
        print(f"Обработанные: {sorted(processed_thermogram_ids)}")
    
    if len(unprocessed_ids) == 0:
        print("✓ Все запрошенные термограммы уже обработаны!")
        return df_combined_all
    
    # 3. ОБРАБОТКА НЕОБРАБОТАННЫХ ТЕРМОГРАММ
    successful_count = 0
    failed_count = 0
    failed_ids = []
    
    for thermogram_id in unprocessed_ids:
        print(f"\n{'='*40}")
        print(f"Термограмма {thermogram_id}")
        print(f"{'='*40}")
        
        try:
            # Проверяем существование файлов
            original_file = f'{original_base_path}/thermogram_{thermogram_id}.npy'
            filtered_file = f'{filtered_base_path}/thermogram_{thermogram_id}_filtered.npy'
            
            if not os.path.exists(original_file):
                print(f"✗ Оригинальный файл не найден: {original_file}")
                failed_ids.append(thermogram_id)
                failed_count += 1
                continue
                
            if not os.path.exists(filtered_file):
                print(f"✗ Фильтрованный файл не найден: {filtered_file}")
                failed_ids.append(thermogram_id)
                failed_count += 1
                continue
            
            print(f"✓ Файлы найдены")
            print(f"  Оригинал: {original_file}")
            print(f"  Фильтр: {filtered_file}")
            
            # Загрузка данных
            frames = np.load(original_file)
            filtered_video = np.load(filtered_file)
            print(f"✓ Загружено: {frames.shape[0]} кадров, {frames.shape[1]}x{frames.shape[2]}")
            
            # ПАЙПЛАЙН ОБРАБОТКИ
            print("┌─ Запуск пайплайна обработки...")
            
            # 3.1 Детекция брызг и зоны сварки
            print("├─ Детекция брызг и зоны сварки...")
            spatters_list = detect_spatters_video(filtered_video)
            welding_boxes, welding_ellipses = detect_welding_zone_video(frames)
            
            # 3.2 Фильтрация брызг
            print("├─ Фильтрация брызг (мертвая зона)...")
            cleaned_spatters = apply_dead_zone(spatters_list, welding_ellipses, dead_zone_radius=50)
            
            print("├─ Фильтрация брызг (коридор)...")
            cleaned_spatters = apply_fixed_corridor_incremental(cleaned_spatters, welding_ellipses, corridor_width=35, lookahead=25)
            
            print("├─ Фильтрация горячими полосами...")
            cleaned_spatters, _ = filter_spatters_by_hot_stripes_video_oriented(
                frames=frames,
                cleaned_spatters_list=cleaned_spatters,
                threshold_low=500,
                threshold_high=1200,  
                min_area=35,
                circular_thr=4.0,
                angle_smooth=0.3,
                initial_angle=-3*np.pi/4,
                show_progress=False
            )
            
            print("├─ Удаление статичных брызг...")
            cleaned_spatters, _ = remove_static_spatters(cleaned_spatters, max_shift=5, min_frames=10)
            
            # 3.3 Расчет статистик
            print("├─ Расчет статистик...")
            df_n_spatters = create_spatters_count_dataframe(cleaned_spatters, thermogram_id=thermogram_id)
            df_spatters_temperature = calculate_spatters_median_temperature(
                cleaned_spatters=cleaned_spatters, 
                frames=frames, 
                threshold_temp=420
            )
            df_welding_temperature = calculate_welding_zone_temperature_threshold(
                welding_ellipses=welding_ellipses, 
                frames=frames, 
                threshold_temp=1000, 
                radius=50, 
                thermogram_id=thermogram_id
            )
            
            print("├─ Трекинг брызг...")
            cleaned_spatters_id, df_stats = track_spatters_with_ids(
                cleaned_spatters=cleaned_spatters,
                frame_width=frames.shape[2],
                frame_height=frames.shape[1],
                min_movement_distance=5,
                min_movement_frames=3
            )
            
            # 3.4 Объединение результатов
            print("├─ Объединение результатов...")
            df_combined = df_n_spatters.merge(df_spatters_temperature, on='frame_number', how='outer')
            df_combined = df_combined.merge(df_welding_temperature, on='frame_number', how='outer')
            df_combined = df_combined.merge(df_stats, on='frame_number', how='outer')
            
            df_combined['thermogram_id'] = thermogram_id
            
            # 3.5 Добавление к общим данным
            if df_combined_all.empty:
                df_combined_all = df_combined
            else:
                df_combined_all = pd.concat([df_combined_all, df_combined], ignore_index=True)
            
            # 4. СОХРАНЕНИЕ ПРОГРЕССА
            processed_thermogram_ids.append(thermogram_id)
            successful_count += 1
            
            # Сохраняем после каждой успешной термограммы
            df_combined_all.to_pickle(df_save_path)
            with open(ids_save_path, 'wb') as f:
                pickle.dump(processed_thermogram_ids, f)
            
            print(f"└─ ✓ Термограмма {thermogram_id} успешно обработана")
            print(f"   Всего обработано: {len(processed_thermogram_ids)}")
            print(f"   Всего строк данных: {len(df_combined_all)}")
            
        except Exception as e:
            print(f"└─ ✗ Ошибка при обработке термограммы {thermogram_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_ids.append(thermogram_id)
            failed_count += 1
            continue
    
    # 5. ФИНАЛЬНАЯ ОБРАБОТКА
    if not df_combined_all.empty:
        df_combined_all = df_combined_all.sort_values(['thermogram_id', 'frame_number']).reset_index(drop=True)
        
        # Финальное сохранение
        df_combined_all.to_pickle(df_save_path)
        with open(ids_save_path, 'wb') as f:
            pickle.dump(processed_thermogram_ids, f)
    
    # 6. ИТОГИ
    print(f"\n{'='*50}")
    print("ИТОГИ ОБРАБОТКИ")
    print(f"{'='*50}")
    print(f"Запрошено термограмм: {len(thermogram_ids)}")
    print(f"Успешно обработано: {successful_count}")
    print(f"Ошибок/пропусков: {failed_count}")
    
    if failed_count > 0:
        print(f"Термограммы с ошибками: {sorted(failed_ids)}")
    
    if not df_combined_all.empty:
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"  Всего строк данных: {len(df_combined_all)}")
        print(f"  Термограмм в данных: {df_combined_all['thermogram_id'].nunique()}")
        print(f"  Диапазон кадров: {df_combined_all['frame_number'].min()} - {df_combined_all['frame_number'].max()}")
        
        # Статистика по термограммам
        thermogram_stats = df_combined_all.groupby('thermogram_id').size()
        print(f"  Строк по термограммам:")
        for thermogram_id, count in thermogram_stats.items():
            print(f"    - Термограмма {thermogram_id}: {count} кадров")
    
    print(f"\nФайлы сохранены в '{save_dir}':")
    print(f"  • df_combined_all.pkl - данные ({len(df_combined_all)} строк)")
    print(f"  • processed_thermogram_ids.pkl - список ID ({len(processed_thermogram_ids)} ID)")
    
    return df_combined_all


def load_saved_processing_results(save_dir='processing_progress'):
    """
    Загрузить сохраненные результаты обработки
    
    Args:
        save_dir: директория с сохраненными результатами
    
    Returns:
        tuple: (DataFrame с данными, список обработанных ID)
        Если файлы не найдены, возвращает (None, [])
    """
    
    df_save_path = f'{save_dir}/df_combined_all.pkl'
    ids_save_path = f'{save_dir}/processed_thermogram_ids.pkl'
    
    if not os.path.exists(df_save_path):
        print("✗ Файл с данными не найден")
        return None, []
    
    try:
        # Загрузка данных
        df_combined_all = pd.read_pickle(df_save_path)
        
        # Загрузка списка ID
        processed_thermogram_ids = []
        if os.path.exists(ids_save_path):
            with open(ids_save_path, 'rb') as f:
                processed_thermogram_ids = pickle.load(f)
        
        print(f"✓ Результаты загружены:")
        print(f"  • Данные: {len(df_combined_all)} строк")
        print(f"  • Обработанные термограммы: {len(processed_thermogram_ids)} шт")
        
        if not df_combined_all.empty and 'thermogram_id' in df_combined_all.columns:
            unique_ids = sorted(df_combined_all['thermogram_id'].unique())
            print(f"  • Термограммы в данных: {unique_ids}")
        
        return df_combined_all, processed_thermogram_ids
        
    except Exception as e:
        print(f"✗ Ошибка при загрузке результатов: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def load_saved_processing_results(save_dir='processing_progress'):
    """
    Загрузить сохраненные результаты обработки
    
    Args:
        save_dir: директория с сохраненными результатами
    
    Returns:
        tuple: (DataFrame с данными, список обработанных ID)
        Если файлы не найдены, возвращает (None, [])
    """
    
    df_save_path = f'{save_dir}/df_combined_all.pkl'
    ids_save_path = f'{save_dir}/processed_thermogram_ids.pkl'
    
    if not os.path.exists(df_save_path):
        print("✗ Файл с данными не найден")
        return None, []
    
    try:
        # Загрузка данных
        df_combined_all = pd.read_pickle(df_save_path)
        
        # Загрузка списка ID
        processed_thermogram_ids = []
        if os.path.exists(ids_save_path):
            with open(ids_save_path, 'rb') as f:
                processed_thermogram_ids = pickle.load(f)
        
        print(f"✓ Результаты загружены:")
        print(f"  • Данные: {len(df_combined_all)} строк")
        print(f"  • Обработанные термограммы: {len(processed_thermogram_ids)} шт")
        
        if not df_combined_all.empty and 'thermogram_id' in df_combined_all.columns:
            unique_ids = sorted(df_combined_all['thermogram_id'].unique())
            print(f"  • Термограммы в данных: {unique_ids}")
        
        return df_combined_all, processed_thermogram_ids
        
    except Exception as e:
        print(f"✗ Ошибка при загрузке результатов: {e}")
        import traceback
        traceback.print_exc()
        return None, []