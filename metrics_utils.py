import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }


def display_metrics_table(metrics_dict):
    st.markdown("### 📈 Evaluation Metrics")
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Score"])
    st.table(metrics_df.style.format({"Score": "{:.2f}"}))


def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.markdown("### 📊 Classification Report")
    st.dataframe(report_df.style.format("{:.2f}"))


def plot_confusion_matrix(y_true, y_pred):
    st.markdown("### 🔢 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Fake'],
                yticklabels=['Genuine', 'Fake'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


def plot_metrics_bar(metrics_dict):
    st.markdown("### 📊 Visual Performance Overview")
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = list(metrics_dict.keys())
    scores = list(metrics_dict.values())
    bars = ax.bar(metrics, scores, color='skyblue', edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Metrics")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.2f}", ha='center')
    st.pyplot(fig)
