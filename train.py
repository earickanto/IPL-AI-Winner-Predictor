import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Model
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv("data/ball_by_ball_ipl.csv")

# Remove unnecessary column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Use only second innings
df = df[df['Innings'] == 2]

# Remove rows with missing values
df = df.dropna()

print("CLEAN DATASET SHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())



#---------------------------------- 


# Create result column
df['result'] = np.where(
    df['Bat Second'] == df['Winner'],
    1,
    0
)

# Show result distribution
print("\nRESULT DISTRIBUTION:")
print(df['result'].value_counts())

# Preview
print("\nTARGET PREVIEW:")
print(df[['Bat Second', 'Winner', 'result']].head(10))  




#----------------------------------

# Current Run Rate
df['current_run_rate'] = (
    df['Innings Runs'] * 6
) / (120 - df['Balls Remaining'])

# Required Run Rate
df['required_run_rate'] = (
    df['Runs to Get'] * 6
) / df['Balls Remaining']

# Wickets Left
df['wickets_left'] = 10 - df['Innings Wickets']

# Select important features
model_df = df[[
    'Bat First',
    'Bat Second',
    'Venue',
    'Runs to Get',
    'Balls Remaining',
    'wickets_left',
    'current_run_rate',
    'required_run_rate',
    'result'
]]

# Remove infinity values
model_df = model_df.replace([np.inf, -np.inf], np.nan)

# Drop nulls
model_df = model_df.dropna()

# Show dataset
print("\nFINAL MODEL DATASET:")
print(model_df.head())

print("\nMODEL DATASET SHAPE:")
print(model_df.shape) 



#---------------------------------- 

# Features
X = model_df.drop('result', axis=1)

# Target
y = model_df['result']

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)     


#----------------------------------

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN SHAPE:")
print(X_train.shape)

print("\nTEST SHAPE:")
print(X_test.shape)    


#----------------------------------


# Categorical columns
cat_features = [
    'Bat First',
    'Bat Second',
    'Venue'
]


#------------------------------

# Create model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    verbose=100
)    



#------------------------------- 

# Train model
model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

print("\nMODEL TRAINED SUCCESSFULLY")  


#-------------------------------

# Predict probabilities
y_prob = model.predict_proba(X_test)

# Predict classes
y_pred = model.predict(X_test)

print("\nPREDICTIONS:")
print(y_pred[:10])

print("\nPROBABILITIES:")
print(y_prob[:5])  


#-------------------------------

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log Loss
loss = log_loss(y_test, y_prob)

print("\nMODEL EVALUATION")

print("Accuracy:", accuracy)

print("Log Loss:", loss)    


#-------------------------------

import pickle

# Save model
pickle.dump(
    model,
    open('models/ipl_win_predictor.pkl', 'wb')
)

print("\nMODEL SAVED SUCCESSFULLY")