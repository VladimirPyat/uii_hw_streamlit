import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def load_data_from_csv(file_path, columns_to_drop=None):
    """Загружает данные из CSV и фильтрует ненужные столбцы."""
    df = pd.read_csv(file_path)
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
    print('Данные успешно загружены')
    return df

def prepare_data(df, y_column):
    """Подготавливает данные для обучения и нормализует их."""
    X = df.drop(columns=[y_column]).values
    y = df[y_column].values
    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    return X_scaled, y


def create_linear_regression_model(input_dim, learning_rate=0.001):
    """Создает модель линейной регрессии с использованием Keras."""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Настройка оптимизатора с заданной скоростью обучения
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, model_save_path='best_model.h5'):
    """Обучает модель и сохраняет лучшую."""
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')
    return model.fit(X_train, y_train, 
             validation_data=(X_val, y_val), 
             epochs=epochs, 
             batch_size=batch_size, 
             callbacks=[checkpoint])
    
def load_trained_model(model_save_path):
    """Загружает модель из файла .h5."""
    return load_model(model_save_path)


def predict_with_model(model, input_data):
    """Принимает на вход модель и массив входных данных, возвращает предсказание."""
    # Убедитесь, что input_data является numpy массивом
    input_data = np.array(input_data)
    # Проверка что размерность входных данных совпадает с размерностью модели
    expected_shape = model.input_shape  
    if input_data.shape[1:] != expected_shape[1:]:
        raise ValueError(f"Неверная форма входных данных. Ожидалась форма {expected_shape}, но получена форма {input_data.shape}.")
    # Изменяем форму для подхода к входному слою
    input_data = input_data.reshape(1, -1)  
    prediction = model.predict(input_data)
    return prediction[0][0]  # Возвращаем одно значение предсказания

boston_model = load_trained_model('best_model.h5')

if __name__ == '__main__':
    # обучение модели и сохранение лучшей
    boston_df = load_data_from_csv('train.csv', columns_to_drop=['ID', 'black'])
    X, y = prepare_data(boston_df, 'medv')
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    boston_model = create_linear_regression_model(X_train.shape[1])
    history = train_model(boston_model, X_train, y_train, X_test, y_test, epochs=300, batch_size=16)
    
    




