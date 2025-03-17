from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import dice_ml
from dice_ml.utils import helpers  # Utility functions

# Step 1: Load dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone", "ForeignWorker", "Target"]
dataset = pd.read_csv(data_url, delim_whitespace=True, names=columns)

dataset.head()

target = dataset['Target']
train_dataset, test_dataset, y_train , y_test = train_test_split(dataset, target, test_size=0.2, random_state=42, stratify=target)
x_train = train_dataset.drop(columns=['Target'], axis=1)
x_test = test_dataset.drop(columns=['Target'], axis=1)

d = dice_ml.Data(dataframe=dataset, continuous_features=['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable'], outcome_name='Target')
numerical = ['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)
    ])

clf = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', RandomForestClassifier())])

model = clf.fit(x_train, y_train)

print("IF THIS IS THE LAST THING PRINTED, THEN GOD HELP US ALL")
m = dice_ml.Model(model=model, backend="sklearn")

exp = dice_ml.Dice(d, m, method="random")

query_instance = x_train[1:2]
cf = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_range=None,
                                  desired_class=1,
                                  permitted_range=None, features_to_vary="all")

# WHY DOESNT THIS WORK FOR ME
#x_test_2 = x_test[test_dataset["Target"] == 2].head(1)


# Generate counterfactuals targeting 1 (THIS DOES NOT WORK)
#cf = exp.generate_counterfactuals(
#    x_test,
#    total_CFs=3,
#    desired_class=1,
#    features_to_vary=[col for col in x_test.columns if col != "Purpose"]
#)
print("Feature importance below")
print("Local")
imp = exp.local_feature_importance(query_instance, cf_examples_list=cf.cf_examples_list)
print(imp.local_importance)

print("Global")
cobj = exp.global_feature_importance(x_train[0:10], total_CFs=10, posthoc_sparsity_param=None)
print(cobj.summary_importance)


pd.set_option('display.max_columns', None)
cf.visualize_as_dataframe(show_only_changes=True)
cf.to_json()
print("Debugging below")
print(model.predict(x_test[0:5]))
print(test_dataset.loc[x_test.index, "Target"])

cf.to_json()