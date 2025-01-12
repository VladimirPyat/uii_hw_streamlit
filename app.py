import streamlit as st
import numpy as np
import pandas as pd

from models.keras_boston import predict_with_model, boston_model


# Заголовок приложения
st.title("Прогноз стоимости недвижимости Бостона")

# Создание пользовательских вводов
crim = st.number_input("Crim (уголовная статистика)", min_value=0.0, value=0.0)
zn = st.number_input("ZN (зоны жилой застройки)", min_value=0.0, value=0.0)
indus = st.number_input("Indus (доля земли, занятой промышленностью)", min_value=0.0, value=0.0)
chas = st.selectbox("Chas (близость к реке Чарльз)", options=[0, 1], index=0)
nox = st.number_input("NOX (концентрация оксидов азота)", min_value=0.0, value=0.0)
rm = st.number_input("RM (среднее количество комнат)", min_value=0.0, value=0.0)
age = st.number_input("AGE (доля построенных до 1940 года)", min_value=0.0, value=0.0)
dis = st.number_input("DIS (дистанция до центров занятости)", min_value=0.0, value=0.0)
rad = st.number_input("RAD (индекс доступности радиусных дорог)", min_value=0, value=1)
tax = st.number_input("TAX (налог на имущество)", min_value=0, value=0)
ptratio = st.number_input("PTRATIO (соотношение учеников к учителям)", min_value=0.0, value=0.0)
lstat = st.number_input("LSTAT (доля населения с низким статусом)", min_value=0.0, value=0.0)

input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, lstat]])

# Кнопка для расчета стоимости
if st.button("Рассчитать стоимость"):
    prediction = predict_with_model(boston_model, input_data)
    st.success(f"Предсказанная стоимость недвижимости 1000$: {prediction:.2f}")


