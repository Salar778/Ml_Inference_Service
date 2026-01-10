import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



Model_Dir = "model"
Model_Path = os.path.join(Model_Dir, "iris_model.joblib")

def main():
    os.makedirs(Model_Dir, exist_ok=True)
    
    data = load_iris()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline ([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    artifact = {
        "model" : pipeline,
        "target_names" : list(data.target_names),
        "feature_names" : list(data.feature_names),
        "accuracy" : float(acc)
        }
    
    joblib.dump(artifact, Model_Path)
    print(f"Model saved to: {Model_Path}")
    print(f"Test accuracy : {acc:.4f}")
    
if __name__ == "__main__":
    main()


    

    
    
    
