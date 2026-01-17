import numpy as np
import scipy.signal as sig
import cv2
from tqdm import tqdm
import os

def min_loc_LoG(img, k_size = 9, sigma = 1.8):
    """
    Perform min-loc-LoG filtering of grayscale image img
    Sungho K. Min-local-LoG Filter for Detecting Small Targets in 
    Cluttered Background // Electronics Letters. 
    – 2011. – Vol. 47. – № 2. – P. 105-106. DOI: 10.1049/el.2010.2066.

    sigma - std of gaussian
    k_size - size of kernel
    """
    x = np.arange(k_size).reshape(1, k_size)
    y = np.arange(k_size).reshape(k_size, 1)
    # generate fE (positive X)
    fE = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fE[fE > 0] = fE[fE > 0] / fE[fE > 0].sum()
    fE[fE < 0] = fE[fE < 0] / (-fE[fE < 0].sum())
    # generate fS (positive Y)
    fS = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fS[fS > 0] = fS[fS > 0] / fS[fS > 0].sum()
    fS[fS < 0] = fS[fS < 0] / (-fS[fS < 0].sum())
    # generate fW
    x = - np.fliplr(x)
    fW = (1 - (x**2) / (sigma**2)) * np.exp(- (x**2) / (2*(sigma**2)))
    fW[fW > 0] = fW[fW > 0] / fW[fW > 0].sum()
    fW[fW < 0] = fW[fW < 0] / (-fW[fW < 0].sum())
    # generate fN
    y = - np.flipud(y)
    fN = (1 - (y**2) / (sigma**2)) * np.exp(- (y**2) / (2*(sigma**2)))
    fN[fN > 0] = fN[fN > 0] / fN[fN > 0].sum()
    fN[fN < 0] = fN[fN < 0] / (-fN[fN < 0].sum())
    # perform 2D convolution with kernels
    def move(img, x, y):
        move_matrix = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv2.warpAffine(img, move_matrix, dimensions)

    Ie = sig.convolve2d(move(img, 4, 0), fE, mode = "same")
    Is = sig.convolve2d(move(img, 0, 4), fS, mode = "same")
    Iw = sig.convolve2d(move(img, -4, 0), fW, mode = "same")
    In = sig.convolve2d(move(img, 0, -4), fN, mode = "same")
    f = np.dstack((Ie, Is, Iw, In))
    fmap = np.min(f, axis = 2)
    #return (fmap / fmap.max() * 255).astype(np.uint8)
    return fmap

def min_loc_LoG_video(frames: np.ndarray, k_size: int = 9, sigma: float = 1.8) -> np.ndarray:
    """
    Apply min-loc-LoG filtering to each frame with progress bar
    
    Args:
        frames: numpy array of shape (N, H, W) - video frames
        k_size: size of kernel
        sigma: std of gaussian
    
    Returns:
        filtered_frames: numpy array of shape (N, H, W) - filtered video
    """
    filtered_frames = []
    
    for i in tqdm(range(frames.shape[0]), desc="Processing frames"):
        frame_result = min_loc_LoG(frames[i], k_size, sigma)
        filtered_frames.append(frame_result)
    
    return np.array(filtered_frames)


def create_filtered_videos_for_ids(thermogram_ids, 
                                   input_base_path='../Data/NumPy_convert',
                                   output_base_path='../Data/Filtered',
                                   k_size=9, sigma=1.8,
                                   overwrite_all=False):
    """
    Создает фильтрованные версии видео для списка термограмм
    
    Args:
        thermogram_ids: список номеров термограмм
        input_base_path: путь к исходным файлам
        output_base_path: путь для сохранения фильтрованных файлов
        k_size: размер ядра для фильтра LoG
        sigma: параметр sigma для фильтра LoG
        overwrite_all: если True, перезаписывает все существующие файлы без запроса
    """
    
    processed = 0
    skipped = 0
    errors = []
    
    for thermogram_id in thermogram_ids:
        print(f"\n{'='*50}")
        print(f"Обработка термограммы {thermogram_id}")
        print(f"{'='*50}")
        
        try:
            # Формируем пути
            file_path = f'{input_base_path}/thermogram_{thermogram_id}.npy'
            output_path = f'{output_base_path}/thermogram_{thermogram_id}_filtered.npy'
            
            # Проверяем существование исходного файла
            if not os.path.exists(file_path):
                print(f"✗ Исходный файл не найден: {file_path}")
                errors.append(f"Термограмма {thermogram_id}: исходный файл не найден")
                continue
            
            # Проверяем существование выходного файла
            if os.path.exists(output_path) and not overwrite_all:
                # Если не overwrite_all, запрашиваем подтверждение
                answer = input(f"Файл {output_path} уже существует. Перезаписать? (y/n/a для всех): ").strip().lower()
                
                if answer == 'a':
                    overwrite_all = True
                    print("Будут перезаписаны все существующие файлы")
                elif answer != 'y':
                    print(f"✓ Пропускаем термограмму {thermogram_id} (файл уже существует)")
                    skipped += 1
                    continue
            
            print(f"✓ Загружаем: {file_path}")
            frames = np.load(file_path)
            print(f"  Размер исходных данных: {frames.shape}")
            
            # Обработка видео
            print(f"  Применяем фильтр LoG (k_size={k_size}, sigma={sigma})...")
            filtered_video = min_loc_LoG_video(frames, k_size=k_size, sigma=sigma)
            print(f"  Размер фильтрованных данных: {filtered_video.shape}")
            
            # Сохранение результата
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, filtered_video)
            
            processed += 1
            print(f"✓ Термограмма {thermogram_id} успешно обработана")
            print(f"  Сохранено в: {output_path}")
            
        except Exception as e:
            error_msg = f"Термограмма {thermogram_id}: {str(e)}"
            print(f"✗ Ошибка: {error_msg}")
            errors.append(error_msg)
            continue
    
    # Итоги обработки
    print(f"\n{'='*50}")
    print("ИТОГИ ОБРАБОТКИ")
    print(f"{'='*50}")
    print(f"Всего термограмм в списке: {len(thermogram_ids)}")
    print(f"Успешно обработано: {processed}")
    print(f"Пропущено (уже существуют): {skipped}")
    print(f"Ошибок: {len(errors)}")
    
    if errors:
        print("\nОшибки при обработке:")
        for error in errors:
            print(f"  - {error}")
    
    return processed, skipped, errors