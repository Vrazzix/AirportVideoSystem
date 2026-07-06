# Airport Video System

Веб-приложение для анализа видео аэропортового обслуживания: детекция колодок у колес, людей, позы человека, событий посадки и статуса обслуживания.

## Запуск

```bash
cd AirportVideoSystem_final/project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python web_app.py
```

После запуска откройте `http://localhost:5000`.

## Модели

Репозиторий содержит код приложения. Веса моделей не рекомендуется хранить в GitHub.

Положите модели в `AirportVideoSystem_final/project/models/` или загрузите их через окно управления моделями в веб-интерфейсе.

Ожидаемые имена по умолчанию:

- `models/BestBoots_v2.pt`
- `models/person.pt`
- `models/BestPose.pt`
- `models/door.pt` (опционально)

Также поддерживаются `.pt`, `.onnx`, `.engine`, `.tflite`, `.pb` и пары OpenVINO `.xml` + `.bin`.

## Структура

- `AirportVideoSystem_final/project/` - приложение Flask.
- `modules/` - обработка кадров, события, трекинг, SLAM и визуализация.
- `templates/` - веб-интерфейс.
- `sources/` - вспомогательные модули для SLAM.
- `requirements.txt` - зависимости для запуска приложения.

Локальные датасеты, ноутбуки, видео, результаты экспериментов и текст диплома разложены по отдельным папкам и исключены из Git.
