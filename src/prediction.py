import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *


def make_prediction(text):
    test_data = pd.DataFrame(data={'text': [text]})
    test_data = test_data['text'].apply(filter_text)
    test_data = test_data.apply(drop_superfuluos)

    tf_idf_vec = TfidfVectorizer(use_idf=True,  
                            smooth_idf=True,  
                            ngram_range=(1,1))
    tf_idf_data = tf_idf_vec.fit_transform(test_data) 
    tf_idf_frame = pd.DataFrame(tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names_out())

    xgb = joblib.load('/home/ejenliya/Projects/FotoCaptions/models/xgboost_clf.pkl')
    features = xgb.get_booster().feature_names
    valid_features = set(tf_idf_frame.columns).intersection(set(features))

    frame_for_prediction = pd.DataFrame(np.random.randint(0, 1, size=(test_data.shape[0], len(features))), columns=features).astype(float)

    for i in range(tf_idf_frame.shape[0]):
        for col in valid_features:
            try:
                frame_for_prediction.loc[i][col] = tf_idf_frame.loc[i][col]
            except Exception as ex:
                print(ex)

    preds = xgb.predict_proba(frame_for_prediction)

    return preds

