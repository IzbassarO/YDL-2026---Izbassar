# Aqyl Jol — CV Perception Layer (Day 3)

**YDL 2026 · Applied Deep Learning · Day 3 Lab ("Build Something with the Right Architecture")**

Мини-версия **компьютерного зрения** проекта [Aqyl Jol](https://orcid.org/0009-0002-4224-3249)
(AI цифровой двойник городского транспорта; победитель Meta Llama Accelerator Track 3). **Три CNN**,
всё обучено/запущено локально на MacBook M1 (MPS) — для YDL summer school, без облака и продакшна:

| # | Модель | Задача | Архитектура | Выход |
|---|--------|--------|-------------|-------|
| 1 | **Качество дороги** | детекция повреждений | YOLOv8 | боксы: pothole / crack / manhole |
| 2 | **Состояние дороги** | погода | ResNet18 transfer learning | fog / rain / snow / sand |
| 3 | **Транспорт** | детекция + тип | YOLOv8 | боксы: car / bus / truck / … |

Архитектура = **CNN** (картинки = сетки → свёртки). Два детектора рисуют типизированные боксы, классификатор тегает погоду.

## Результаты
- **Качество дороги (YOLOv8):** боксы pothole/crack/manhole, mAP50 ≈ **0.40** (~12 эпох на MPS; ямы/люки — уверенно).
- **Состояние дороги (ResNet18):** val-accuracy **~0.82** (snow 0.93, sand 0.87; fog/rain путаются). Grad-CAM ок.
- **Транспорт (YOLOv8):** ~**50 FPS** на M1 (MPS), боксы car/bus/bicycle в движении (Aqyl Jol: 60–125 FPS).

## Файлы
| Файл | Что делает |
|------|-----------|
| `prepare_data.py` | DAWN (плоские фото) → `train/val` по классам погоды из имён файлов |
| `train_road.py` | ResNet18 transfer learning → `data/road_resnet18.pt` + confusion matrix |
| `gradcam_road.py` | Grad-CAM панели → `artifacts/road_gradcam.png` |
| `yolo_video.py` | YOLOv8 на видео → `artifacts/traffic_detected.mp4` + кадры + метрики |
| `aqyl_jol_cv_demo.ipynb` | **главный артефакт**: research-ноутбук со сквозной историей |

## Артефакты
- `artifacts/road_confusion.png` — матрица ошибок классификатора дороги
- `artifacts/road_gradcam.png` — куда смотрит сеть
- `artifacts/traffic_detected.mp4` + `traffic_frame_*.jpg` — детекция машин
- `data/road_resnet18.pt` — обученная модель

## Данные
- **DAWN** (Kaggle `shuvoalok/dawn-dataset`, ~1027 фото, CC-BY-NC-SA): реальные дорожные сцены,
  классы fog/rain/snow/sand. Фолбэк: [Mendeley](https://data.mendeley.com/datasets/766ygrbt8y/3).
- Видео: Intel IoT `sample-videos` (уличная сцена с трафиком).

## Воспроизвести
```bash
# окружение: conda env nn-lab (Python 3.11) + torch 2.12 (MPS), ultralytics, grad-cam
cd "3 week/3 day/aqyl_jol_cv"
python prepare_data.py     # DAWN -> train/val (нужен скачанный data/images)
python train_road.py       # -> data/road_resnet18.pt (~1-2 мин на MPS)
python gradcam_road.py      # -> artifacts/road_gradcam.png
python yolo_video.py        # -> artifacts/traffic_detected.mp4
python build_notebook.py    # -> aqyl_jol_cv_demo.ipynb (исполняется)
```
Запуск моделей — в env **nn-lab** (`/opt/anaconda3/envs/nn-lab/bin/python`).

## Ограничения / дальше
- DAWN без «потоп/грязь»; Aqyl Jol использовал свои снимки Астаны + **сегментацию** (по пикселям).
- Детектор машин — pretrained COCO; Aqyl Jol дообучал на 14k фото Астаны, 23 класса.
- Next: разморозить последний блок ResNet; дообучить YOLO на локальных типах; перейти к сегментации дороги.
