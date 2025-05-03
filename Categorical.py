import dice_ml
import pandas as pd
import numpy as np
from dice_ml.utils import helpers  # Utility functions
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Step 1: Load dataset
data_path = "statlog+german+credit+data/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone", "ForeignWorker", "Target"]
dataset = pd.read_csv(data_path, delim_whitespace=True, names=columns)



dataset.head()

# Step 2: Preprocess the dataset
target = dataset['Target']
train_dataset, test_dataset, y_train , y_test = train_test_split(dataset, target, test_size=0.2, random_state=42, stratify=target)
x_train = train_dataset.drop(columns=['Target'], axis=1)
x_test = test_dataset.drop(columns=['Target'], axis=1)

# Step 3: Encode categorical features
d = dice_ml.Data(dataframe=dataset, continuous_features=['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable'], outcome_name='Target')
numerical = ['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable']
categorical = x_train.columns.difference(numerical)

# Define preprocessing pipelines for numerical and categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define preprocessing pipelines for numerical features
transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)
    ])

# Apply transformations to training and testing data
clf = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', RandomForestClassifier())])

# Fit the model
model = clf.fit(x_train, y_train)


# Step 4: Use DiCE for Counterfactuals
m = dice_ml.Model(model=model, backend="sklearn")

exp = dice_ml.Dice(d, m, method="random")

#instance_index = dataset[dataset["Target"] == 2].index[0]
instance_index = np.random.choice(dataset[dataset["Target"] == 2].index)

# Select a negative instance (bad credit case)
query_instance = x_test.loc[[instance_index]]
cf = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_range=None,
                                  desired_class="opposite",
                                  permitted_range=None, features_to_vary="all")

# Step 5: Visualize the results
print("Feature importance below")
print("Local")
imp = exp.local_feature_importance(query_instance, cf_examples_list=cf.cf_examples_list)
print(imp.local_importance)

print("Global")
cobj = exp.global_feature_importance(x_train[0:10], total_CFs=10, posthoc_sparsity_param=None)
print(cobj.summary_importance)

# Step 6: Visualize the counterfactuals, altering dataframe settings
pd.set_option('display.max_columns', None)
cf.visualize_as_dataframe(show_only_changes=True)
cf.to_json()

print("Debugging below")
print(model.predict(x_test[0:5]))
print(test_dataset.loc[x_test.index, "Target"])

cf.to_json()