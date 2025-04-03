import dice_ml
import pandas as pd
from dice_ml.utils import helpers  # Utility functions
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# Step 1: Load dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment", "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone", "ForeignWorker", "Target"]
dataset = pd.read_csv(data_url, delim_whitespace=True, names=columns)

target = dataset['Target']
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=42, stratify=target)
x_train = train_dataset.drop(columns=['Target'], axis=1)
x_test = test_dataset.drop(columns=['Target'], axis=1)

d = dice_ml.Data(dataframe=dataset, continuous_features=['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable'], outcome_name='Target')
numerical = ['Duration', 'CreditAmount', 'InstallmentRate', 'Residence', 'Age', 'ExistingCredits', 'NumLiable']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

transformations = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ]
)

x_train_transformed = transformations.fit_transform(x_train)
x_test_transformed = transformations.transform(x_test)

# Convert the numpy array to a dense DataFrame
x_train_transformed_dense = pd.DataFrame(x_train_transformed, columns=transformations.get_feature_names_out())
x_test_transformed_dense = pd.DataFrame(x_test_transformed, columns=transformations.get_feature_names_out())

# Build the TensorFlow model
model = Sequential([
    Input(shape=(x_train_transformed_dense.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_transformed_dense, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Wrap the trained model with DiCE
m = dice_ml.Model(model=model, backend="TF2")

exp = dice_ml.Dice(d, m, method="random")

instance_index = dataset[dataset["Target"] == 2].index[0]
query_instance = x_test.loc[[instance_index]]
query_instance_transformed = transformations.transform(query_instance)

# Convert the query instance to a dense format
query_instance_transformed_dense = pd.DataFrame(query_instance_transformed, columns=transformations.get_feature_names_out())

cf = exp.generate_counterfactuals(query_instance_transformed_dense, total_CFs=10, desired_range=None,
                                  desired_class="opposite",
                                  permitted_range=None, features_to_vary="all")

print("Feature importance below")
print("Local")
imp = exp.local_feature_importance(query_instance_transformed_dense, cf_examples_list=cf.cf_examples_list)
print(imp.local_importance)

print("Global")
cobj = exp.global_feature_importance(x_train_transformed_dense[0:10], total_CFs=10, posthoc_sparsity_param=None)
print(cobj.summary_importance)

pd.set_option('display.max_columns', None)
cf.visualize_as_dataframe(show_only_changes=True)
cf.to_json()

print("Debugging below")
print(model.predict(x_test_transformed_dense[0:5]))
print(test_dataset.loc[x_test.index, "Target"])

cf.to_json()