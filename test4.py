import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn


st.set_page_config(page_title= "test for web app", layout= "centered")


def func(x):
    dict2 = {0: "very low", 1: "low", 2: "high", 3: "very high"}
    return dict2[x]

dict1 = {"very low":0,"low":1,"high":2,"very high":3}


@st.cache_resource
def load():
    try:
        load_job = joblib.load("model_and_columns_name.joblib")
        return load_job["model"], load_job["columns_name"]
    #I know: load_job["columns_name"] = model.feature_name_in_
    except FileNotFoundError:
        st.error("joblib file doesn't exist")
        st.stop()
    except Exception as e:
        st.error(f"error in load file: {e}")
        st.stop()
        
        
model, feature_names = load()



def get_prediction(input_columns: dict):
    try:
        X_web = pd.DataFrame([input_columns])[feature_names]
        pred = model.predict(X_web)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            class_index = list(model.classes_).index(pred)
            prob = model.predict_proba(X_web)[0][class_index]
            
        return pred, prob
    except Exception as e:
        st.error(f"error in model process: {e}")
        st.stop()




st.title("سیستم پیش بینی بیماری")

st.markdown("<br>", unsafe_allow_html=True)

form = st.form(key= "get_features")

with form:
    
    col1, col2 = st.columns(2)
    
    with col1:
        sex = int(st.selectbox("sex", [0,1], format_func= lambda x:'M' if x == 1 else 'F'))
        cp = int(st.select_slider("cp", [0,1,2,3], format_func= func, value= 1))
        thall = int(dict1[st.select_slider("thall", ["very low","low","high","very high"], value= "low")])
    with col2:
        chol = int(st.number_input("chol", 50, 700, 250, step= 1))
        age = int(st.slider("age" , 20, 100, 50, step=1)    )

    submitted = st.form_submit_button(label= "submit features")


if submitted:
    input_columns = {'age': age, 'chol': chol, 'cp': cp, 'thall': thall, 'sex': sex}
    prediction, probability = get_prediction(input_columns)
    
    label1 = {1: "patient", 0: "ok"}
    label2 = label1.get(prediction, "unknown")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("prediction is:")
        if prediction == 1:
            st.error(label2)
        elif prediction == 0:
            st.success(label2)
        else:
            st.warning(label2)
    with col2:
        st.subheader("probability is:", help= f"probability for {label2}")
        if probability is not None:
            st.info(f"{probability*100:.1f}%")
        else:
            st.warning("probability is not available")






