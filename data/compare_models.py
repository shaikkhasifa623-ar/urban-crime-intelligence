
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the historical NCRB dataset
df = pd.read_csv("urban_crime_fullstack/data/dstrIPC_2013.csv")

# Use crime-category counts as features.
# Exclude total crimes because it is calculated from crime counts.
# Exclude district/state identifiers and year from the target inputs.
target_column = "TOTAL IPC CRIMES"

excluded = [
    "STATE/UT", "DISTRICT", "YEAR", target_column
]

X = df.drop(columns=excluded)
y = (df[target_column] >= df[target_column].median()).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, random_state=42
    ),
}

results = []

for name, model in models.items():
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        model
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 3),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 3),
        "F1-score": round(f1_score(y_test, predictions, zero_division=0), 3),
    })

results_df = pd.DataFrame(results).sort_values(
    by="F1-score", ascending=False
)

print("\nMODEL COMPARISON — HISTORICAL 2013 DATA")
print(results_df.to_string(index=False))

results_df.to_csv(
    "urban_crime_fullstack/model_comparison.csv", index=False
)
print("\nSaved results to model_comparison.csv")
