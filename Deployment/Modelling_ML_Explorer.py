import pandas as pd
import numpy as np
import pickle

import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


def prepare_data_split(df, test_size, random_state, target_feature):
    '''
    Fuunction Split a dataset(df) in the ratio 60%/20%/20% == Train/Validation/Test
    
    return the train, Validation and Test dataset with their corresponding targer variable
    '''
    df_full_train, df_test = train_test_split(df, test_size= test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=random_state)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)
    
    df_train.fillna(0, inplace = True)
    df_val.fillna(0, inplace = True)
    df_test.fillna(0, inplace = True)
    df_full_train.fillna(0, inplace = True)
    
    y_train = np.log1p(df_train[target_feature].values)
    y_val = np.log1p(df_val[target_feature].values)
    y_test = np.log1p(df_test[target_feature].values)
    y_full_train = np.log1p(df_full_train[target_feature].values)
    
    del df_train[target_feature]
    del df_val[target_feature]
    del df_test[target_feature]
    del df_full_train[target_feature]
    
    return df_train, y_train, df_val, y_val, df_test, y_test, df_full_train, y_full_train


df = pd.read_csv("model_data_2.csv")

df.dropna(subset = ["s1quant_sold"], inplace = True)

num_cols = df.select_dtypes("number").columns
cat_cols = df.select_dtypes("object").columns
    
# Fill missing values
for col in num_cols:
    df[col].fillna(0, inplace = True)

for c in cat_cols:
    nam = 'Unknown_%s' % c
    df[c].fillna(nam, inplace = True)


le = LabelEncoder()
encodelist = []
for c in df.select_dtypes("object").columns:
    #perform label encoding Dataset
    df[c] = le.fit_transform(df[c])
    
    # Get the classes and their corresponding encoded values
    classes = le.classes_
    encoded_values = le.transform(classes)
    encodelist.append({'classes_%s' %c:classes, 'encoded_values_%s' %c:encoded_values})

#Saving the LabelEncoder as "le_1.pkl"
with open("le_1.pkl", "wb") as f:
    pickle.dump(le, f)
    
print("LabelEncoder saved as 'le_1.pkl'"
    
# Feature selected via the RFE training tagged as "final_features"
final_features = ['Country', 'Region', 'fsystem1', 'tenure1', 'yearsuse1', 'rentplot1', 's1start', 's1end', 'seas1nam', 's1plant_data',
                  's1land_area', 's1quant_harv', 's1consumed', 's1livestock', 's1lost', 's1market', 's1quant_sold', 's1crop_val', 
                  's1no_seed', 'pc1', 'nyieldc1', 's1irrig1', 's1irrig2', 's1irrig3', 's1irrig4', 's1pest', 's1wat1', 's1wat2', 's1wat3',
                  's1wat4', 's1wat5', 'costkgfert', 'costkgpest', 'distsmktkm', 'distsmkthr', 'distpmktkm', 'distpmkthr', 'transport', 
                  'cost1crop', 'cost2crop', 'cost3crop', 'cost5crop', 'farmingexperience', 'ad711', 'ad718', 'ad7111', 'ad7116', 'ad7120',
                  'ad732', 'ad742', 'ad7511', 'ad7610', 'ad7613', 'ad7624']

df = df[final_features]

# Setting up Validation framework
test_size = 0.2
random_state = 1
target = "s1quant_sold"
X_train, y_train, X_val, y_val, X_test, y_test, X_full_train, y_full_train = prepare_data_split(df,
                                                                                                test_size,
                                                                                                random_state, 
                                                                                                target)

# LightGBM Regressor gave the best model during model selection and hence used for modelling script
      
# Hyper tuning LightGBM Regressor model

# Define hyperparameters to tune
param_grid = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [30, 50, 70, 90, 110, 130, 150, 170, 190],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Initialize GridSearchCV with the LightGBM model and parameter grid
grid_search = GridSearchCV(estimator=lgb.LGBMRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Perform the hyperparameter search on the training data
grid_search.fit(X_train, y_train)


# Get the best hyperparameters and model for Lightgbm 

best_params = grid_search.best_params_
best_lightgbm_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_lightgbm_model.predict(X_full_train)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_full_train, y_pred))
print("Best LightGBM Model RMSE:", rmse)


# Save your lightgbm model as `"model_gmb_1.pkl and the labelEncode as le_gmb_1.pkl"`
with open("model_gmb_1.pkl", "wb") as f:
    pickle.dump(best_lightgbm_model, f)
    
