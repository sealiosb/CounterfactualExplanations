import pandas as pd
import dice_ml
from dice_ml.utils import helpers  # Utility functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings", "Employment",
           "InstallmentRate", "PersonalStatus", "OtherDebtors", "Residence", "Property", "Age",
           "OtherInstallment", "Housing", "ExistingCredits", "Job", "NumLiable", "Telephone",
           "ForeignWorker", "Target"]
df = pd.read_csv(data_url, delim_whitespace=True, names=columns)


label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

index_target_2 = df[df["Target"] == 2].index.tolist()

if index_target_2:
    first_bad_index = index_target_2[0]  # First occurrence
    next_index = first_bad_index + 1     # Next row

    if next_index < len(df):
        next_row = df.iloc[next_index]


df["Target"] = df["Target"].apply(lambda x: 1 if x == 1 else 0)

# Train-test split
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Use DiCE for Counterfactuals
backend = dice_ml.Model(model=model, backend="sklearn")
data_interface = dice_ml.Data(dataframe=df, continuous_features=X.columns.tolist(), outcome_name='Target')
dice = dice_ml.Dice(data_interface, backend)

# Select a negative instance (bad credit case)
negative_instance_index = df[df["Target"] == 0].index[0]  # Select first "bad" instance
instance = pd.DataFrame(X_test[negative_instance_index].reshape(1, -1), columns=X.columns)

# Generate Counterfactuals
cf = dice.generate_counterfactuals(instance, total_CFs=10, desired_range=None,
                                  desired_class="opposite",
                                  permitted_range=None, features_to_vary="all")
cf.visualize_as_dataframe()
