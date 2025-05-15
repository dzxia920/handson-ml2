import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

# Load the dataset
housing = pd.read_csv('./datasets/housing/housing.csv')

# split train and test data
# housing_train, housing_test = train_test_split(housing, test_size=0.2, random_state=42)
# stratify 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
for train_index, test_index in sss.split(housing, housing['income_cat']):
    housing_train = housing.loc[train_index]
    housing_test = housing.loc[test_index]
for set_ in (housing_train, housing_test):
    set_.drop('income_cat', axis=1, inplace=True)


# 分割标签和特征
housing_train_set = housing_train.drop('median_house_value', axis=1)
housing_test_set = housing_test.drop('median_house_value', axis=1)
housing_train_labels = housing_train['median_house_value'].copy()
housing_test_labels = housing_test['median_house_value'].copy()

# # 分为数值和文本值
housing_num = housing_train_set.select_dtypes(include=[np.number])
housing_cat = housing_train_set[['ocean_proximity']]

#　pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

# 组合数值和文本值
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, list(housing_num)),
    ('cat', OneHotEncoder(), ['ocean_proximity']),
])

housing_prepared = full_pipeline.fit_transform(housing_train_set)

# 选择模型
model = SVR()

# 网格搜索
param_grid = [
    {'kernel': ['linear', 'rbf'], 'C': [3, 10], 'gamma': [0.01, 0.05]},
]

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_train_labels)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)

final_model = grid_search.best_estimator_

# 交叉验证
scores = cross_val_score(final_model, housing_prepared, housing_train_labels, n_jobs=5,
                           scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(lin_rmse_scores)

final_model.fit(housing_prepared, housing_train_labels)

# 选择好最终的模型后再进行测试
housing_test_prepared = full_pipeline.transform(housing_test_set)
final_predictions = final_model.predict(housing_test_prepared)
final_mse = mean_squared_error(housing_test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final RMSE:", final_rmse)