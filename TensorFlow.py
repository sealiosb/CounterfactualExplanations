import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import dice_ml

# Load dataset
data_path = '/Users/owen/PycharmProjects/CounterfactualExplanations/statlog+german+credit+data/german.data'
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment",
           "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age",
           "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone",
           "ForeignWorker", "Target"]
dataset = pd.read_csv(data_path, delim_whitespace=True, names=columns)

# Encode target column to binary (0 and 1)
label_encoder = LabelEncoder()
dataset['Target'] = label_encoder.fit_transform(dataset['Target'])

# Split dataset into train and test sets
target_column = "Target"
numerical_features = ["Duration", "CreditAmount", "InstallmentRate", "Residence", "Age", "ExistingCredits", "NumLiable"]
target = dataset[target_column]
train_dataset, test_dataset, y_train, y_test = train_test_split(
    dataset, target, test_size=0.2, random_state=42, stratify=target)

x_train = train_dataset.drop(columns=[target_column], axis=1)
x_test = test_dataset.drop(columns=[target_column], axis=1)

# Define categorical features as all columns except numerical ones
categorical_features = x_train.columns.difference(numerical_features).tolist()

# Define transformations for preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

transformations = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Transform training and testing data
x_train_transformed = transformations.fit_transform(x_train)
x_test_transformed = transformations.transform(x_test)

# Build and train the TensorFlow model
input_dim = x_train_transformed.shape[1]
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output probabilities for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_transformed, y_train.values, epochs=10, batch_size=32, validation_split=0.2)

# Generate counterfactuals
numerical_transformed = [f'num__{col}' for col in numerical_features]
transformed_dataset = pd.DataFrame(
    x_train_transformed,
    columns=transformations.get_feature_names_out()
)
transformed_dataset[target_column] = y_train.values

d = dice_ml.Data(
    dataframe=transformed_dataset,
    continuous_features=numerical_transformed,
    outcome_name=target_column
)
m = dice_ml.Model(model=model, backend="TF2")
exp = dice_ml.Dice(d, m, method="random")

# Select a valid instance for counterfactual generation
instance_index = dataset[dataset[target_column] == 1].index[0]
query_instance = pd.DataFrame([dataset.iloc[instance_index].drop(target_column)])
query_instance_transformed = transformations.transform(query_instance)
query_instance_transformed_dense = pd.DataFrame(
    query_instance_transformed,  # Already a dense numpy array
    columns=transformations.get_feature_names_out()
)

cf = exp.generate_counterfactuals(
    query_instance_transformed_dense,
    total_CFs=10,
    desired_class="opposite",  # Adjust to match the opposite class
    features_to_vary="all"
)

# Visualize counterfactuals
if cf.cf_examples_list[0].final_cfs_df.empty:
    print("No counterfactuals found. Try adjusting the parameters or model.")
else:
    pd.set_option('display.max_columns', None)
    print(cf.visualize_as_dataframe(show_only_changes=True))