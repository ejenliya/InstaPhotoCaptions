import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import yaml

with open('../FotoCaptions/config/config.yaml') as f:
    temps = yaml.safe_load(f)
random_forest, xgboost, svm = temps['random_forest'], temps['xgboost'], temps['svm']


train_data = pd.read_csv('../FotoCaptions/data/prepared_data.csv')
train_data.info()
train_target = train_data.pop('popularity')


train_data_rsmpld, target_rsmpld = ADASYN().fit_resample(train_data, train_target)

tree = DecisionTreeClassifier(max_depth=100)
tree.fit(train_data_rsmpld, target_rsmpld)
preds = tree.predict(train_data_rsmpld)
print('Tree accuracy: ', accuracy_score(preds, target_rsmpld))

feature_importances = pd.Series(tree.feature_importances_, index=train_data.columns).sort_values(ascending=False)
X_train, X_valid, y_train, y_valid = train_test_split(train_data_rsmpld, target_rsmpld, test_size=0.2, random_state=12)


'''xgb = XGBClassifier()
xgb_params = {'n_estimators': np.arange(xgboost['min_estimators'], xgboost['max_estimators'], xgboost['estimators_step']),
         'max_depth': np.arange(xgboost.min_depth, xgboost.max_depth)}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='precision', n_jobs=-1)
xgb_grid.fit(X_train[feature_importances[:xgboost['num_features']].index], y_train)

best_xgb = xgb_grid.best_estimator_
preds = best_xgb.predict(X_valid[feature_importances[:xgboost['num_features']].index])
print('XGB Accuracy: ', accuracy_score(y_valid, preds))
print('XGB Precision: ', precision_score(y_valid, preds))
print('XGB Recall: ', recall_score(y_valid, preds))
print('XGB F1-score: ', f1_score(y_valid, preds))

joblib.dump(best_xgb, '../FotoCaptions/models/xgboost_clf.pkl')'''


rf = RandomForestClassifier()
rf_params = {'n_estimators': np.arange(random_forest['min_estimators'], random_forest['max_estimators'], random_forest['estimators_step']),
          'max_depth': np.arange(random_forest['min_depth'], random_forest['max_depth'], random_forest['depth_step'])}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='precision', n_jobs=-1)
rf_grid.fit(X_train[feature_importances[:random_forest['num_features']].index], y_train) #

best_rf = rf_grid.best_estimator_
preds = best_rf.predict(X_valid[feature_importances[:random_forest['num_features']].index])
print('RF Accuracy: ', accuracy_score(y_valid, preds))
print('RF Precision: ', precision_score(y_valid, preds))
print('RF Recall: ', recall_score(y_valid, preds))
print('RF F1-score: ', f1_score(y_valid, preds))

joblib.dump(best_rf, '../FotoCaptions/models/random_forest_clf.pkl')


'''svc = SVC()
svc_params = {'C': np.arange(svm['min_C'], svm['max_C'], svm['C_step']),
         'kernel': svm['kernel']}

svc_grid = GridSearchCV(svc, svc_params, cv=5, scoring='precision', n_jobs=-1)
svc_grid.fit(X_train[feature_importances[:svm['num_features']].index], y_train)

best_svc = svc_grid.best_estimator_
preds = best_svc.predict(X_valid[feature_importances[:svm['num_features']].index])
print('SVC Accuracy: ', accuracy_score(y_valid, preds))
print('SVC Precision: ', precision_score(y_valid, preds))
print('SVC Recall: ', recall_score(y_valid, preds))
print('SVC F1-score: ', f1_score(y_valid, preds))

joblib.dump(best_svc, '../FotoCaptions/models/svm_clf.pkl')'''