import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/train_IPL.csv")

print(df.shape)

# Remove ties and no results
df = df[
    df['result_type'].isna()
]

# Match-level info
matches = df.groupby('Match ID').first().reset_index()

print(matches.shape)

print(matches.head()) 


#-----------------------------------

# First innings total
innings1 = df[df['Innings'] == 1]

innings1_total = innings1.groupby('Match ID')['Innings Runs'].max()

# Second innings total
innings2 = df[df['Innings'] == 2]

innings2_total = innings2.groupby('Match ID')['Innings Runs'].max()

# Second innings wickets
innings2_wickets = innings2.groupby('Match ID')['Innings Wickets'].max()

# Add totals to matches dataframe
matches['innings1_runs'] = matches['Match ID'].map(innings1_total)

matches['innings2_runs'] = matches['Match ID'].map(innings2_total)

matches['innings2_wickets'] = matches['Match ID'].map(innings2_wickets)

# Preview
print(matches[[
    'Match ID',
    'Bat First',
    'Bat Second',
    'match_won_by',
    'innings1_runs',
    'innings2_runs',
    'innings2_wickets'
]].head())    


#----------------------------------

# Function to create 4-class labels
def create_label(row):

    # Team A won
    if row['match_won_by'] == row['Bat First']:

        margin = row['innings1_runs'] - row['innings2_runs']

        if margin <= 20:
            return 'A_small'
        else:
            return 'A_big'

    # Team B won
    else:

        wickets_left = 10 - row['innings2_wickets']

        if wickets_left <= 5:
            return 'B_small'
        else:
            return 'B_big'


# Apply labels
matches['label'] = matches.apply(create_label, axis=1)

# Show label distribution
print("\nLABEL DISTRIBUTION:")
print(matches['label'].value_counts())

# Preview
print("\nLABEL PREVIEW:")
print(matches[[
    'Match ID',
    'Bat First',
    'Bat Second',
    'match_won_by',
    'label'
]].head(10))    


#----------------------------------


# Feature selection
model_df = matches[[
    'Bat First',
    'Bat Second',
    'Venue',
    'toss_winner',
    'toss_decision',
    'innings1_runs',
    'label'
]]

# Drop nulls
model_df = model_df.dropna()

print("\nMODEL DATA:")
print(model_df.head())

print("\nMODEL SHAPE:")
print(model_df.shape)  


#----------------------------------

# Features
X = model_df.drop('label', axis=1)

# Target
y = model_df['label']

print("\nX SHAPE:")
print(X.shape)

print("\ny SHAPE:")
print(y.shape)  


#----------------------------------


from sklearn.model_selection import train_test_split

# Split
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


from catboost import CatBoostClassifier

# Categorical features
cat_features = [
    'Bat First',
    'Bat Second',
    'Venue',
    'toss_winner',
    'toss_decision'
]

# Create model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    verbose=100
)   


#----------------------------------

# Train model
model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

print("\nMULTI-CLASS MODEL TRAINED")   


#----------------------------------


from sklearn.metrics import log_loss

# Predict probabilities
y_prob = model.predict_proba(X_test)

# Log Loss
loss = log_loss(y_test, y_prob)

print("\nLOG LOSS:")
print(loss)    


#---------------------------------- 

# Load public leaderboard matches
public_matches = pd.read_csv("data/public_lb_matches.csv") 


print(public_matches.columns)

# Load sample submission
submission = pd.read_csv("data/sample_submission.csv")

print("\nPUBLIC MATCHES:")
print(public_matches.head())

print("\nSUBMISSION FORMAT:")
print(submission.head())   



#-----------------------------------

# Create prediction dataframe
predict_df = public_matches[[
    'team_a',
    'team_b',
    'venue',
    'toss_winner',
    'toss_decision'
]].copy()

# Rename columns to match training
predict_df.columns = [
    'Bat First',
    'Bat Second',
    'Venue',
    'toss_winner',
    'toss_decision'
]

# Estimated innings1 score
predict_df['innings1_runs'] = 180

print("\nPREDICT DATA:")
print(predict_df.head())

print("\nPUBLIC MATCH COLUMNS:")
print(public_matches.columns)


#----------------------------------- 

# Fill missing values
predict_df = predict_df.fillna("Unknown")

# Predict probabilities
pred_probs = model.predict_proba(predict_df)


print("\nPREDICTED PROBABILITIES:")
print(pred_probs[:5])  


#------------------------------- 

# Create prediction dataframe
pred_df = pd.DataFrame(
    pred_probs,
    columns=['A_small', 'A_big', 'B_small', 'B_big']
)

# Add default probabilities for remaining 5 matches
extra_rows = pd.DataFrame({
    'A_small': [0.25]*5,
    'A_big': [0.25]*5,
    'B_small': [0.25]*5,
    'B_big': [0.25]*5
})

# Combine
pred_df = pd.concat([pred_df, extra_rows], ignore_index=True)

# Fill submission
submission[['A_small', 'A_big', 'B_small', 'B_big']] = pred_df

# Save submission
submission.to_csv("submission.csv", index=False)

print("\nSUBMISSION FILE CREATED") 


import pickle

# Save trained model
pickle.dump(
    model,
    open('models/ipl_win_predictor.pkl', 'wb')
)

print("\nMODEL SAVED SUCCESSFULLY")
