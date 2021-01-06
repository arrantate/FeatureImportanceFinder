import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class StatusMessages:
    def __init__(self):
        self.message_list = []

    def add(self, message_to_add):
        self.message_list.append(message_to_add)

    def messages(self):
        return "\n".join(self.message_list)


status = StatusMessages()
complete = False

st.title("Feature Importance Finder")
st.markdown(
    """Created by [@arrantate](https://twitter.com/arrantate). **[Buy me a coffee?](
    https://www.buymeacoffee.com/arrantate)**"""
)
st.write("This application uses a random forest classifier to find the importance of your features given a .csv file "
         "and a "
         "target.")
st.write("Note that the app currently assumes no NaN values and does not assign dummies.  For now, data must be clean "
         "and of type int or float. Remember that since this uses a classification model the target cannot be of type "
         "float.")

uploaded_file = st.file_uploader("Choose a file to get started", type="csv")
if uploaded_file is not None:

    with st.spinner(text='Uploading file...'):
        data = pd.read_csv(uploaded_file)
        status.add("- File uploaded  :heavy_check_mark:")

    st.markdown("***")

    target_choices = list(data.columns)
    float_features = list(data.select_dtypes(include=[np.float]).columns)
    for feature in float_features:
        if feature in target_choices:
            target_choices.remove(feature)
    target = st.selectbox("Select Target", target_choices)

    if st.button("Find Importance"):

        with st.spinner(text='Assigning training data...'):
            y = data[target]
            X = data.drop([target], axis=1)
            status.add("- Training data assigned  :heavy_check_mark:")

        with st.spinner(text='Fitting model to training data...'):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            status.add("- Model fitted to training data  :heavy_check_mark:")

        with st.spinner(text='Calculating importance'):
            importances = list(model.feature_importances_)
            feature_importances = [(feature, importance) for feature, importance in zip(X.columns, importances)]

            # Sorting by most important first
            feature_importances = pd.DataFrame(
                sorted(feature_importances, key=lambda x: x[1], reverse=True),
                columns=['Feature', 'Importance']
            ).set_index('Feature')

            status.add("- Importance calculated  :heavy_check_mark:")

            complete = True

st.markdown(status.messages())

if complete:
    st.table(feature_importances)
    max = feature_importances.idxmax(axis=0)[0]
    min = feature_importances.idxmin(axis=0)[0]
    st.markdown(f'Most important feature: **{max}**  \nLeast important feature: **{min}**')
