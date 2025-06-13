import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载第一个文件的模型
model = joblib.load('MLP.pkl')

# 使用的特征名称（来自你刚确认的第二个文件）
feature_ranges = {
    "Age": {"type": "numerical", "min": 12, "max": 72, "default": 50},
    "Symptom Duration": {"type": "numerical", "min": 1, "max": 360, "default": 25},
    "Flexion": {"type": "numerical", "min": 15, "max": 60, "default": 40},
    "Extension": {"type": "numerical", "min": 15, "max": 50, "default": 36},
    "Rotation": {"type": "numerical", "min": 35, "max": 80, "default": 50},
    "Spring test-pain": {"type": "categorical", "options": [0, 1, 2, 3]},
    "Muscle tightness": {"type": "categorical", "options": [0, 1, 2, 3]},
    "Exacerbation on Flexion": {"type": "categorical", "options": [0, 1]},
    "Exacerbation on Extension": {"type": "categorical", "options": [0, 1]},
}

label_mapping = {
    "Spring test-pain": {0: "无压痛", 1: "单节段", 2: "2-3节段", 3: "广泛压痛"},
    "Muscle tightness": {0: "无", 1: "单部位", 2: "2-3部位", 3: "广泛紧张"},
    "Exacerbation on Flexion": {0: "否", 1: "是"},
    "Exacerbation on Extension": {0: "否", 1: "是"},
}

# Streamlit 应用标题
st.title("MLP模型预测手法治疗获益")
st.header("请输入以下特征:")

# 输入字段（模仿第二个文件）
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        labels = label_mapping.get(feature, {v: str(v) for v in properties["options"]})
        value = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
            format_func=lambda x: labels[x]
        )
    feature_values.append(value)

# 整合输入
features = np.array([feature_values])
feature_names = list(feature_ranges.keys())

# 预测与解释
if st.button("预测"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测文本
    if predicted_class == 1:
        advice = f"模型预测：您可能从手法治疗中获益（预测概率为 {probability:.2f}%）"
    else:
        advice = f"模型预测：您可能不从手法治疗中获益（预测概率为 {probability:.2f}%）"

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, advice, fontsize=16, ha='center', va='center', fontname='SimHei', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    df_input = pd.DataFrame([feature_values], columns=feature_names)
    explainer = shap.Explainer(lambda x: model.predict_proba(x), df_input)
    shap_values = explainer(df_input)
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
