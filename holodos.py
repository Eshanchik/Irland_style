import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Загрузка и обработка изображения
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Определение объектов с помощью YOLOv3
def detect_objects(image):
    model = YOLO('yolov3.pt')  # Загрузка предобученной модели YOLOv3
    results = model(image)  # Распознавание объектов
    return results

# Отрисовка результатов
def plot_objects(results, image):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Показ исходного изображения

    # Обработка результатов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Координаты бокса
            conf = box.conf[0]  # Уверенность
            label = result.names[int(box.cls[0])]  # Класс объекта
            
            # Отрисовка прямоугольника и подписи
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='blue', facecolor='none', lw=2))
            ax.text(x1, y1, f'{label} {conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

    plt.show()

# Главная функция
def main(image_path):
    # Шаг 1: Загрузка изображения
    image = load_image(image_path)

    # Шаг 2: Определение объектов с помощью YOLOv3
    results = detect_objects(image)

    # Шаг 3: Визуализация объектов
    plot_objects(results, image)

# Запуск программы
image_path = ""  # Укажите путь к изображению
main(image_path)
