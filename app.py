import streamlit as st
import pickle
import numpy as np

# モデルを読み込む
with open("SVM.pkl", "rb") as f:
    model = pickle.load(f)  # そのままPipelineオブジェクトとして読み込む

scaler = model.named_steps["scaler"]
svm = model.named_steps["clf"]

# タイトル
st.title("Brugada Syndrome Risk Prediction")

# ユーザー入力フォーム
st.sidebar.header("Input Features")

# 入力フィールド
r_J_interval = st.sidebar.number_input("r-J interval in V1 (ms)", min_value=0.0, max_value=200.0, step=0.1)
QRS_V6 = st.sidebar.number_input("QRS duration in V6 (ms)", min_value=0.0, max_value=200.0, step=0.1)
T_peak_to_T_end = st.sidebar.number_input("T-peak-to-T-end interval (ms)", min_value=0.0, max_value=200.0, step=0.1)
age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=1)

syncope = st.sidebar.radio("Syncope", [0, 1])
frag_QRS = st.sidebar.radio("Fragmented QRS", [0, 1])
ER_presence = st.sidebar.radio("Presence of ER", [0, 1])


# 予測ボタン
if st.sidebar.button("Predict"):
    # 入力データを配列に変換
    input_data = np.array([[r_J_interval, syncope,frag_QRS, ER_presence,T_peak_to_T_end, QRS_V6,age]])

    # 予測
    #prediction = svm.predict(input_data)
    #probability = model.predict_proba(input_data)[0][1]  # 1（疾患あり）の確率
    #probability = svm.predict_proba(input_data)[0][1] * 100
    probability = model.decision_function(input_data)

    # 結果表示
    st.subheader("Prediction Result")
    #if prediction[0] == 1:
    #    st.error(f"⚠️ High Risk (Probability: {probability:.2%})")
    #else:
    #    st.success(f"✅ Low Risk (Probability: {probability:.2%})")
    # 確率の数値を表示
    if float(probability) <= 0:
        st.write("Probability of Brs: 0")
    else:
        st.write(f"Probability of Brs: {probability}")

