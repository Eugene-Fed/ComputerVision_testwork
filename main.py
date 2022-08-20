# импорты
from datetime import timedelta
import cv2
import numpy as np
import os
import json

SAVING_FRAMES = 60                                      # сохраняем каждый 15-й кадр
WORK_DIRECTORY = "D:\Downloads\ComputerVision_test"
VIDEO_FILE_NAME = "7.mp4"


def main(video_file):
    filename, _ = os.path.splitext(video_file)          # разделяем имя файла на название и расширение
    if not os.path.isdir(filename):                     # создаем папку по названию видео файла
        os.mkdir(filename)
    cap = cv2.VideoCapture(video_file)                  # читаем видео файл

    # цикл по всем кадрам, начиная с 15 (нужно начинать с 14 т.к. индекс 0-based) до конца файла, через каждые 15 кадров
    for frame_no in range(SAVING_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), SAVING_FRAMES):
        # единица в качестве первого параметра - это CV_CAP_PROP_POS_FRAMES
        # или '0-based index of the frame to be decoded/captured next', т.е. метод set(1, int) задает кадр для чтения
        cap.set(1, frame_no);
        is_read, frame = cap.read()     # is_read = True если такой кадр существует, frame - это объект кадра
        cv2.imwrite(os.path.join(filename, "frame{}.jpg".format(frame_no)), frame)  # сохраняем кадр в файл


if __name__ == '__main__':
    main('{}\{}'.format(WORK_DIRECTORY, VIDEO_FILE_NAME))

