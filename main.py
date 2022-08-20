# импорты
from datetime import timedelta
import cv2
import numpy as np
import os
import json

SAVING_FRAMES = 15                                  # сохраняем каждый 15-й кадр
WORK_DIRECTORY = "D:\Downloads\ComputerVision_test"
VIDEO_FILE_NAME = "7.mp4"


def format_timedelta(td):
    """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
    исключая микросекунды и сохраняя миллисекунды"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """Функция, которая возвращает список длительностей, в которые следует сохранять кадры."""
    s = []
    # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используем np.arange () для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def main(video_file):
    filename, _ = os.path.splitext(video_file)          # разделяем имя файла на название и расширение
    if not os.path.isdir(filename):                     # создаем папку по названию видео файла
        os.mkdir(filename)
    cap = cv2.VideoCapture(video_file)                  # читаем видео файл
    fps = cap.get(cv2.CAP_PROP_FPS)                     # получаем FPS видео
    # если SAVING_FRAMES_PER_SECOND выше видео FPS, то установить его равным FPS (тогда сохранится каждый кадр)
    # saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    saving_frames_per_second = fps // SAVING_FRAMES     # вычисляем количество сохраняемых кадров в секунду
    # получить список длительностей для сохранения
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # запускаем цикл
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # выйти из цикла, если нет фреймов для чтения
            break
        # получаем продолжительность, разделив количество кадров на FPS
        frame_duration = count / fps
        try:
            # получить самую раннюю продолжительность для сохранения
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # список пуст, все кадры длительности сохранены
            break
        if frame_duration >= closest_duration:
            # если ближайшая длительность меньше или равна длительности кадра,
            # затем сохраняем фрейм
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame)
            # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров
        count += 1


if __name__ == '__main__':
    main('{}\{}'.format(WORK_DIRECTORY, VIDEO_FILE_NAME))

