#https://github.com/sealiosb/CounterfactualExplanations
#If issues are found with the code, please report them to the author of the code.
# This code generates counterfactual explanations for a TensorFlow model using the DiCE library.


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import dice_ml

# Load the dataset
data_path = "statlog+german+credit+data/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment",
           "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age",
           "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone",
           "ForeignWorker", "Target"]
dataset = pd.read_csv(data_path, delim_whitespace=True, names=columns)

# Encode the target column to binary values (0 and 1)
label_encoder = LabelEncoder()
dataset['Target'] = label_encoder.fit_transform(dataset['Target'])

# Define target column and numerical features
target_column = "Target"
numerical_features = ["Duration", "CreditAmount", "InstallmentRate", "Residence", "Age", "ExistingCredits", "NumLiable"]

# Split the dataset into training and testing sets
target = dataset[target_column]
train_dataset, test_dataset, y_train, y_test = train_test_split(
    dataset, target, test_size=0.2, random_state=42, stratify=target)

# Separate features and target for training and testing
x_train = train_dataset.drop(columns=[target_column], axis=1)
x_test = test_dataset.drop(columns=[target_column], axis=1)

# Identify categorical features (all columns except numerical ones)
categorical_features = x_train.columns.difference(numerical_features).tolist()

# Define preprocessing pipelines for numerical and categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Scale numerical features
])

# Combine preprocessing steps into a single transformer
transformations = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Apply numerical transformations
        ('cat', categorical_transformer, categorical_features)  # Apply categorical transformations
    ]
)

# Apply transformations to training and testing data
x_train_transformed = transformations.fit_transform(x_train)
x_test_transformed = transformations.transform(x_test)

# Build a TensorFlow neural network model
input_dim = x_train_transformed.shape[1]
model = Sequential([
    Input(shape=(input_dim,)),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the transformed training data
model.fit(x_train_transformed, y_train.values, epochs=10, batch_size=32, validation_split=0.2)

# Prepare the dataset for counterfactual generation
numerical_transformed = [f'num__{col}' for col in numerical_features]
transformed_dataset = pd.DataFrame(
    x_train_transformed,
    columns=transformations.get_feature_names_out()
)
transformed_dataset[target_column] = y_train.values

# Create a DiCE data object for counterfactual generation
d = dice_ml.Data(
    dataframe=transformed_dataset,
    continuous_features=numerical_transformed,
    outcome_name=target_column
)

# Create a DiCE model object for the trained TensorFlow model
m = dice_ml.Model(model=model, backend="TF2")

# Initialize the DiCE explainer
exp = dice_ml.Dice(d, m, method="random")

# Select a valid instance for counterfactual generation
instance_index = dataset[dataset[target_column] == 1].index[0]
query_instance = pd.DataFrame([dataset.iloc[instance_index].drop(target_column)])
query_instance_transformed = transformations.transform(query_instance)
query_instance_transformed_dense = pd.DataFrame(
    query_instance_transformed,
    columns=transformations.get_feature_names_out()
)

# Generate counterfactuals for the query instance
cf = exp.generate_counterfactuals(
    query_instance_transformed_dense,
    total_CFs=10,  # Generate 10 counterfactuals
    desired_class="opposite",
    features_to_vary="all"
)

# Visualize the counterfactuals
if cf.cf_examples_list[0].final_cfs_df.empty:
    print("No counterfactuals found. Try adjusting the parameters or model.")
else:
    pd.set_option('display.max_columns', None)  # Display all columns in the output
    print(cf.visualize_as_dataframe(show_only_changes=False))
    print("Changes only below")
    print(cf.visualize_as_dataframe(show_only_changes=True))