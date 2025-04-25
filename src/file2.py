import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException

# Load environment variables
load_dotenv()

# Set MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "Rohit-tipstat")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Rohit-tipstat/mlflow_tutorial.mlflow")

print(f"Accessing as {os.getenv('MLFLOW_TRACKING_USERNAME')}")

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='Rohit-tipstat', repo_name='mlflow_tutorial', mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define RF model parameters
max_depth = 8
n_estimators = 5

# Set or create experiment
try:
    experiment = mlflow.get_experiment_by_name("Experiment - 1")
    if experiment is None:
        mlflow.create_experiment("Experiment - 1")
    mlflow.set_experiment("Experiment - 1")
    print("Experiment set successfully")
except MlflowException as e:
    print(f"Failed to set experiment: {e}")
    raise

# Start MLflow run
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix")
    
    # Save and log plot
    plt.savefig("Confusion_matrix.png")
    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    # Set tags
    mlflow.set_tags({"Author": "Rohit", "Project": "Wine Classifier"})
    
    # Log the model
    mlflow.sklearn.log_model(rf, "Random Forest Model")
    
    print(f"Accuracy: {accuracy}")