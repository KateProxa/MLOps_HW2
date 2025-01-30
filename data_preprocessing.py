import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys


def load_and_preprocess(input_path, output_path):
    # Загрузка данных
    red_wine = pd.read_csv(f"{input_path}/winequality-red.csv", sep=";")
    white_wine = pd.read_csv(f"{input_path}/winequality-white.csv", sep=";")

    # Объединение датасетов и добавление метки типа вина
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'
    wines = pd.concat([red_wine, white_wine], ignore_index=True)

    # Очистка данных: удаление пропущенных значений
    wines.dropna(inplace=True)

    # Кодирование категориального признака (типа вина)
    wines['type'] = wines['type'].apply(lambda x: 1 if x == 'red' else 0)

    # Разделение признаков и целевого признака (качество вина)
    features = wines.drop(columns=['quality'])
    target = wines['quality']

    # Отделение категориального признака перед нормализацией
    wine_type = features['type']
    features_to_normalize = features.drop(columns=['type'])

    # Нормализация признаков
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_to_normalize)

    # Преобразование нормализованных признаков в DataFrame и добавление 'type'
    processed_data = pd.DataFrame(
        features_normalized, columns=features_to_normalize.columns)
    processed_data['type'] = wine_type.reset_index(drop=True)

    # Добавление целевого признака 'quality'
    processed_data['quality'] = target.reset_index(drop=True)

    # Сохранение очищенных и нормализованных данных
    processed_data.to_csv(
        f"{output_path}/cleaned_normalized_wine_data.csv", index=False)


if __name__ == "__main__":
    load_and_preprocess(sys.argv[1], sys.argv[2])
