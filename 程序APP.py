import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os
# 下载字体文件
font_url = 'https://raw.githubusercontent.com/BUCMGYY/Prof-Zhu-Liguo-Traditional-Chinese-Medicine-Decision-Making-Assistance-Model-for-CS-Diagnosis/main/SimHei.ttf'  # 确保这个链接是正确的
font_path = 'SimHei.ttf'

# 检查字体文件是否已存在，若不存在则下载
if not os.path.exists(font_path):
    response = requests.get(font_url)
    with open(font_path, 'wb') as f:
        f.write(response.content)

# 加载保存的随机森林模型
model = joblib.load('MLP.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "年龄": {
        "type": "numerical",
        "min": 12.0,
        "max": 72.0,
        "default": 50.0
    },
    "症状持续时间(天)": {
        "type": "numerical",
        "min": 1.0,
        "max": 360.0,
        "default": 25.0
    },
    "前屈活动度": {
        "type": "numerical",
        "min": 15.0,
        "max": 60.0,
        "default": 40.0
    },
    "后伸活动度": {
        "type": "numerical",
        "min": 15.0,
        "max": 50.0,
        "default": 36.0
    },
    "旋转活动度": {
        "type": "numerical",
        "min": 35.0,
        "max": 80.0,
        "default": 50.0
    },
    "压痛试验": {
        "type": "categorical",
        "options": [0, 1, 2, 3]
    },
    "肌肉紧张程度": {
        "type": "categorical",
        "options": [0, 1, 2, 3]
    },
    "前屈加重": {
        "type": "categorical",
        "options": [0, 1]
    },
    "后伸加重": {
        "type": "categorical",
        "options": [0, 1]
    }
}

label_mapping = {
    "压痛试验": {
        0: "无压痛",
        1: "单节段",
        2: "2-3节段",
        3: "广泛压痛"
    },
    "肌肉紧张程度": {
        0: "无",
        1: "单部位",
        2: "2-3部位",
        3: "广泛紧张"
    },
    "前屈加重": {
        0: "否",
        1: "是"
    },
    "后伸加重": {
        0: "否",
        1: "是"
    }
}

# 动态生成输入项
st.title("手法疗效预测模型")
st.header("请输入以下特征:")
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
        # 获取中文标签映射
        labels = label_mapping[feature]
        value = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
            format_func=lambda x: labels[x]  # 显示中文标签
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 定义类别映射字典
category_mapping = {
    0: "低获益",
    1: "高获益"
}

# 预测与 SHAP 可视化
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取所有类别的概率
    probabilities = predicted_proba * 100

    # 创建预测结果文本
    predicted_class_name = category_mapping[predicted_class]  # 获取中文类别名称
    text = f"模型预测：您接受手法治疗的获益类别为: {predicted_class_name}(截断概率阈值为60.3%），概率分别为：\n"
    for i, prob in enumerate(probabilities):
        category_name = category_mapping[i]  # 获取中文类别名称
        text += f"{category_name}: {prob:.2f}%\n"
    # 加载字体文件
    prop = fm.FontProperties(fname=font_path)
    # 显示预测结果，使用 Matplotlib 渲染指定字体
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontproperties=prop,  # 使用加载的字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 添加字体路径
    fm.fontManager.addfont(font_path)
    # 设置字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
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
