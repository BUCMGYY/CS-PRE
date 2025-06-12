
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载第一个文件的模型
model = joblib.load('MLP.pkl')

# 使用的特征名称（来自你刚确认的第二个文件）
feature_names = [
    "Age", "Symptom Duration", "Flexion", "Extension", "Rotation",
    "Spring test-pain", "Muscle tightness", "Exacerbation on Flexion",
    "Exacerbation on Extension"
]

# Streamlit 应用标题
st.title("MLP模型预测手法治疗获益（保留第二个界面）")

# 输入字段（模仿第二个文件）
age = st.number_input("年龄:", min_value=12, max_value=72, value=50, step=1)
symptom_duration = st.number_input("症状持续时间(天):", min_value=1, max_value=360, value=25, step=1)
flexion = st.number_input("前屈活动度:", min_value=15, max_value=60, value=40, step=1)
extension = st.number_input("后伸活动度:", min_value=15, max_value=50, value=36, step=1)
rotation = st.number_input("旋转活动度:", min_value=35, max_value=80, value=50, step=1)
spring_test_pain = st.selectbox("压痛试验 (0=无压痛, 1=单节段, 2=2-3节段, 3=广泛压痛):", options=[0, 1, 2, 3])
muscle_tightness = st.selectbox("肌肉紧张程度 (0=无, 1=单部位, 2=2-3部位, 3=广泛紧张):", options=[0, 1, 2, 3])
exacerbation_on_flexion = st.selectbox("前屈加重 (0=否, 1=是):", options=[0, 1])
exacerbation_on_extension = st.selectbox("后伸加重 (0=否, 1=是):", options=[0, 1])

# 整合输入
feature_values = [
    age, symptom_duration, flexion, extension, rotation,
    spring_test_pain, muscle_tightness, exacerbation_on_flexion, exacerbation_on_extension
]
features = np.array([feature_values])

# 预测与解释
if st.button("预测"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = f"模型预测：您可能从手法治疗中获益（预测概率为 {probability:.2f}%）"
    else:
        advice = f"模型预测：您可能不从手法治疗中获益（预测概率为 {probability:.2f}%）"

    # 显示预测文字
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, advice, fontsize=16, ha='center', va='center', fontname='SimHei', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # SHAP 可视化（使用第一个文件方式）
    df_input = pd.DataFrame([feature_values], columns=feature_names)
    explainer = shap.Explainer(lambda x: model.predict_proba(x), df_input)
    shap_values = explainer(df_input)

    shap.force_plot(
        base_value=explainer.expected_value[0],
        shap_values=shap_values[0],
        features=df_input.iloc[0],
        feature_names=feature_names,
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
