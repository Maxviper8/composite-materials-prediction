import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Загрузка моделей (без изменений)
model_mod = joblib.load('model_elastic_modulus.pkl')
model_str = joblib.load('model_tensile_strength.pkl')
scaler = joblib.load('scaler.pkl')

model_ratio = tf.keras.models.load_model('model_ratio.h5', compile=False)
scaler_ratio = joblib.load('scaler_ratio.pkl')

# ---- Признаки для прогноза модуля и прочности (11 шт.) ----
feature_names_all = [
    'Соотношение матрица-наполнитель',
    'Плотность, кг/м3',
    'модуль упругости, ГПа',
    'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2',
    'Температура вспышки, С_2',
    'Поверхностная плотность, г/м2',
    'Потребление смолы, г/м2',
    'Угол нашивки, град',
    'Шаг нашивки',
    'Плотность нашивки'
]

# ---- Признаки для рекомендации соотношения (12 шт. в правильном порядке) ----
feature_names_ratio = [
    'Плотность, кг/м3',
    'модуль упругости, ГПа',
    'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2',
    'Температура вспышки, С_2',
    'Поверхностная плотность, г/м2',
    'Модуль упругости при растяжении, ГПа',   # желаемое значение
    'Прочность при растяжении, МПа',           # желаемое значение
    'Потребление смолы, г/м2',
    'Угол нашивки, град',
    'Шаг нашивки',
    'Плотность нашивки'
]

# Функция для прогноза модуля и прочности (без изменений)
def predict_properties(*args):
    data = np.array(args).reshape(1, -1)
    df_input = pd.DataFrame(data, columns=feature_names_all)
    mod_pred = model_mod.predict(df_input)[0]
    str_pred = model_str.predict(df_input)[0]
    return f"Модуль упругости: {mod_pred:.2f} ГПа", f"Прочность: {str_pred:.2f} МПа"

# Функция для рекомендации соотношения (исправлено)
def recommend_ratio(*args):
    # args — 12 значений в порядке feature_names_ratio
    data = np.array(args).reshape(1, -1)
    df_input = pd.DataFrame(data, columns=feature_names_ratio)
    X_scaled = scaler_ratio.transform(df_input)
    ratio_pred = model_ratio.predict(X_scaled)[0][0]
    return f"Рекомендуемое соотношение матрица-наполнитель: {ratio_pred:.3f}"

# Строим интерфейс
with gr.Blocks(title="Прогнозирование свойств композитов") as demo:
    gr.Markdown("# Прогнозирование свойств композиционных материалов")
    
    with gr.Tab("Прогноз модуля и прочности"):
        inputs = []
        for name in feature_names_all:
            inputs.append(gr.Number(label=name, value=0.0))
        
        btn = gr.Button("Рассчитать")
        out_mod = gr.Textbox(label="Модуль упругости")
        out_str = gr.Textbox(label="Прочность при растяжении")
        btn.click(fn=predict_properties, inputs=inputs, outputs=[out_mod, out_str])
    
    with gr.Tab("Рекомендация соотношения матрица-наполнитель"):
        inputs_ratio = []
        for name in feature_names_ratio:
            if name == 'Модуль упругости при растяжении, ГПа':
                inputs_ratio.append(gr.Number(label=name, value=70.0))
            elif name == 'Прочность при растяжении, МПа':
                inputs_ratio.append(gr.Number(label=name, value=2500.0))
            else:
                inputs_ratio.append(gr.Number(label=name, value=0.0))
        
        btn_ratio = gr.Button("Рекомендовать")
        out_ratio = gr.Textbox(label="Результат")
        btn_ratio.click(fn=recommend_ratio, inputs=inputs_ratio, outputs=out_ratio)
    
    gr.Markdown("---\n### Загрузка данных из CSV (опционально)")
    with gr.Row():
        gr.File(label="Загрузите CSV с данными")

demo.launch()