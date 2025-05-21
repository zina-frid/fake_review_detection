# Fake Review Detector
Библиотека **fake_review_detection** для распознавания фиктивных отзывовс применением архитектуры трансформер и методов их оптимизации.

Структутра проекта:
```bash
fakereviewdetector/
├── __init__.py
├── trainer.py                # Основной интерфейс
├── simple_trainer.py         # Базовое дообучение
├── bilstm_trainer.py         # Трансформер + BiLSTM
├── pruning_trainer.py        # Трансформер + pruning
├── models.py                 # Классы моделей
├── utils.py                  # Вспомогательные функции
requirements.txt              # Зависимости
setup.py                      # Установщик библиотеки
```

Доступно три режима обучения моделей: 
- `simple` -- дообучение базовой трансформерной модели без модификаций
- `bilstm` -- дообучение трансформера с дополнительным BiLSTM-слоем
- `pruning` -- обучение трансформера с последующим применением прореживания весов

## Быстрый старт
Для установки и использования библиотеки можно воспользоавться Google Colaboratory, используя среду с графическим процессором.

Установка библиотеки в среду:
```python
!pip install git+https://github.com/zina-frid/fake_review_detection.git
```

Импорт функции обучения
```python
from fakereviewdetector.trainer import train_and_evaluate
```

На вход функции ``train_and_evaluate`` подаются следующие аргументы:

*Обязательные*

`train_df`, `val_df`, `test_df` -- pandas-таблицы с текстами и метками (обязательно наличие колонок text и class)

`model_name` -- название предобученной модели (например, "bert-base-uncased" или "roberta-base")

*Опциональные*

`method` -- один из режимов ("simple", "bilstm", "pruning"), по умолчанию "simple"

`max_len`, `batch_size`, `epochs` -- параметры обучения, имеют дефолтные значения

`pruning_amount` -- только для режима pruning

`output_dir` -- папка, куда сохраняются модели

Функция возвращает метрики на тестовой выборке и продолжительность обучения. Все модели сохраняются в стандартном формате Hugging Face (.bin + config.json) по указанному пути.


Пример подготовки `train_df`, `val_df`, `test_df` для обучения модели (`data` -- датафрейм с размеченными данными):
```python
from sklearn.model_selection import train_test_split
import pandas as pd

train_df, temp = train_test_split(
    data,
    test_size=0.2,
    random_state=1,
    stratify=data["class"]
)

val_df, test_df = train_test_split(
    temp,
    test_size=0.5,
    random_state=1,
    stratify=temp["class"]
)
```

Пример запуска обучения:
```python
metrics, duration = train_and_evaluate(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    model_name="roberta-base",
    method="bilstm",
    max_len=160,
    batch_size=32,
    epochs=3,
    output_dir=FILE_PATH + "/saved_models/bilstm/"
)

print("-- Метрики --")
for metric, value in metrics.items():
    print(f"{metric}: {value:.5f}")

print("-- Время обучения --")
print(duration)
```

Пример вывода результатов обучения:
```bash
-- Метрики --
Accuracy: 0.85000
Macro Precision: 0.85377
Macro Recall: 0.86248
Macro F1 Score: 0.84946
Weighted Precision: 0.86883
Weighted Recall: 0.85000
Weighted F1 Score: 0.85090
-- Время обучения --
2:03
```



Библиотека также предоставляет возможность воспользоваться моделью для предсказания с помощью функции `predict_review` из модуля `utils`. 
Пример загрузки обученной модели:
```python
from fakereviewdetector.utils import predict_review

# Путь к сохранённой модели
model_path = FILE_PATH + "/saved_models/bilstm/"

# Отзыв
review_text = "I searched many websites trying to find an affordable cooler bag to replace one I had bought from QVC many years earlier These are just what the doctor ordered at a reasonable price I live out in the country about miles away from the nearest supermarket so these are an essential for me to be able to buy milk products and meats safely especially in hot summer weather You can really load them down You will not regret this purchase They are very easily cleaned if you have a spill and well constructed to last for a long while"

# Предсказание
predicted_label = predict_review(model_path, review_text)


label_map = {0: "REAL", 1: "FAKE"}
print(f"Отзыв: {review_text}")
print(f"Класс: {label_map[predicted_label]}")
```

Пример вывода предсказания:
```bash
Отзыв: I searched many websites trying to find an affordable cooler bag to replace one I had bought from QVC many years earlier These are just what the doctor ordered at a reasonable price I live out in the country about miles away from the nearest supermarket so these are an essential for me to be able to buy milk products and meats safely especially in hot summer weather You can really load them down You will not regret this purchase They are very easily cleaned if you have a spill and well constructed to last for a long while
Класс: FAKE
```
