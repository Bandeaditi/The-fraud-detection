#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


# In[2]:


df = pd.read_csv('Fraud.csv')
# Assuming df is your DataFrame with the given column names
selected_features = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                      'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

target_column = 'isFraud'


# In[ ]:


# Check if selected_features and target_column exist in the DataFrame
if not all(feature in df.columns for feature in selected_features + [target_column]):
    raise ValueError("Some columns are missing in the DataFrame.")

df_selected = df[selected_features + [target_column]]


# In[ ]:


# One-hot encode categorical column 'type'
df_selected = pd.get_dummies(df_selected, columns=['type'], drop_first=True, prefix='type')


# In[ ]:


# Encode non-numeric columns using label encoding
label_encoder = LabelEncoder()
df_selected['nameOrig'] = label_encoder.fit_transform(df_selected['nameOrig'])
df_selected['nameDest'] = label_encoder.fit_transform(df_selected['nameDest'])


# In[ ]:


# Encoding target column 'isFraud'
label_encoder = LabelEncoder()
df_selected[target_column] = label_encoder.fit_transform(df_selected[target_column])


# In[ ]:


# Extracting features (X) and target (y)
X = df_selected.drop(columns=[target_column])
y = df_selected[target_column]


# In[ ]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')  
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[ ]:


# Initialize RandomForestClassifier (you can adjust hyperparameters)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[ ]:


# Train the model
rf_model.fit(X_train_imputed, y_train)


# In[ ]:


# Make predictions on the test set
y_pred = rf_model.predict(X_test_imputed)


# In[ ]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)


# In[ ]:


import matplotlib.pyplot as plt

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_

# Get the column names (feature names)
feature_names = X.columns

# Sort indices based on feature importances
sorted_indices = feature_importances.argsort()[::-1]


# In[ ]:


# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align="center")
plt.xticks(range(len(feature_importances)), feature_names[sorted_indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Classifier - Feature Importance")
plt.show()


# In[3]:


# Check for missing values in the DataFrame
missing_values = df.isnull().sum()

# Display columns with missing values, if any
columns_with_missing_values = missing_values[missing_values > 0]
if not columns_with_missing_values.empty:
    print("Columns with missing values:")
    print(columns_with_missing_values)
else:
    print("No missing values in the DataFrame.")


# In[ ]:




