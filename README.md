# Feature Importance Finder

Deployed tool can be found here:
https://share.streamlit.io/arrantate/featureimportancefinder/main/FeatureImportance.py

This application uses a random forest classifier to find the importance of your features given a .csv file and a target.

Note that the app currently assumes no NaN values and does not assign dummies. For now, data must be clean and of type int or float. Remember that since this uses a classification model the target cannot be of type float.
