import numpy as np
from models.svm import SVM
from models.xgboost_scratch import XGBoostFromScratch
from xgboost import XGBClassifier
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df):
  # normalization and standardization of numeric features
  numeric_df = df.select_dtypes(include=['number'])
  numeric_columns = numeric_df.columns 
  # scaler = StandardScaler()
  # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

  # Feature Engineering Code
  df['area_to_radius_ratio'] = df['area_mean'] / df['radius_mean']

  df['texture_radius_interaction'] = df['texture_mean'] * df['radius_mean']

  mean_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
  df['mean_features_avg'] = df[mean_features].mean(axis=1)
  df['mean_features_std'] = df[mean_features].std(axis=1)

  df['area_to_perimeter_worst_ratio'] = df['area_worst'] / df['perimeter_worst']

  for feature in ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal_dimension']:
      df[f'{feature}_variation'] = df[f'{feature}_worst'] - df[f'{feature}_mean']

  if 'diagnosis' in df.columns:
    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

  return df

def svm_scratch(df, user): 
  df = preprocess(df)
  user = preprocess(user)

  train = df
  train = train.drop(columns="id")
  X_train = train[train['diagnosis'].notnull()]

  Y_train = X_train['diagnosis']
  X_train = X_train.drop(columns=['diagnosis'])

  X_train = X_train.to_numpy()
  Y_train = Y_train.to_numpy()

  model = SVM(learning_rate=0.001, C=1.0, n_iters=2)
  model.fit(X_train, Y_train)

  y_pred = model.predict(user)
  y_pred = np.where(y_pred == -1, 0, y_pred)
  
  return y_pred 

def xgboost_scratch(df, user):
  df = preprocess(df)
  user = preprocess(user)

  train = df
  train = train.drop(columns="id")
  X_train = train[train['diagnosis'].notnull()]

  Y_train = X_train['diagnosis']
  X_train = X_train.drop(columns=['diagnosis'])

  X_train = X_train.to_numpy()
  Y_train = Y_train.to_numpy()

  model = XGBoostFromScratch(n_estimators=200, max_depth=7, learning_rate=0.1, min_child_weight=3, lambda_=0, gamma=0)
  model.fit(X_train, Y_train)

  y_pred = model.predict(user)
  
  return y_pred 

def xgboost_package(df, user):
  df = preprocess(df)
  user = preprocess(user)

  train = df
  train = train.drop(columns="id")
  X_train = train[train['diagnosis'].notnull()]

  Y_train = X_train['diagnosis']
  X_train = X_train.drop(columns=['diagnosis'])

  model = XGBClassifier(tree_method = "hist", device = "cuda").fit(X_train, Y_train.astype(int))
  model.fit(X_train, Y_train)

  y_pred = model.predict(user)
  
  return y_pred 

def svm_package(df, user):
  df = preprocess(df)
  user = preprocess(user)

  train = df
  train = train.drop(columns="id")
  X_train = train[train['diagnosis'].notnull()]

  Y_train = X_train['diagnosis']
  X_train = X_train.drop(columns=['diagnosis'])

  model = SVC(kernel='linear')
  model.fit(X_train, Y_train)

  y_pred = model.predict(user)
  
  return y_pred 



