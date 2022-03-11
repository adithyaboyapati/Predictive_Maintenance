import streamlit as st
import pandas as pd
import pickle
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://images.rawpixel.com/image_400/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjk2MC1uaW5nLTMwLmpwZw.jpg?s=keWypa5MppkqbZQzS93m6cKbPwxkPHO4DEKbu6kF-cQ");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()


st.title('Failure Prediction')
st.subheader('Lets Predict the Failures of the Engine')

try:
    upload_file=st.file_uploader('Choose a csv')
    if upload_file:
        st.markdown('-----')
        df = pd.read_csv(upload_file)
        df = df.apply(pd.to_numeric)
        scaler = pickle.load(open("models/scalar.pkl", 'rb'))
        X_scaled = scaler.transform(df)
        X_scaled = pd.DataFrame(X_scaled)
        pca = pickle.load(open("models/pca.pkl", 'rb'))
        new_data = pca.transform(X_scaled)
        principal_x = pd.DataFrame(new_data, index=X_scaled.index)
        loaded_model = pickle.load(open("models/svm_model.pkl", 'rb'))
        model = pd.DataFrame(loaded_model.predict(principal_x))
        result = model.replace({0: 'Not Failed', 1: 'Failed'})
        st.dataframe(result)
    download = st.button('Download prediction')
    try:
        if download:

            csv = result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            linko = f'<a href="data:file/csv;base64,{b64}" download="engine_failure_prediction.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)
    except:
        st.write('Please enter a file')
except:
    st.write('please enter a file')


