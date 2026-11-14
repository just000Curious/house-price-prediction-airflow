from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import os
import shutil
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

# Quality Gate Threshold
MIN_R2_THRESHOLD = 0.75

@dag(
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["housing", "mlops", "retraining"]
)
def house_retraining_pipeline():

    @task
    def merge_and_clean_data():
        base_path = "/opt/airflow/data/house_data.csv"
        feedback_path = "/opt/airflow/data/feedback_submissions.csv"
        
        df_base = pd.read_csv(base_path)
        
        if os.path.exists(feedback_path):
            df_feedback = pd.read_csv(feedback_path)
            print(f"Loaded {len(df_feedback)} feedback rows.")
            df_combined = pd.concat([df_base, df_feedback], ignore_index=True)
        else:
            print("No feedback file found; using base dataset only.")
            df_combined = df_base
            
        df_combined = df_combined.dropna(subset=["MedHouseVal"])
        print(f"Total clean rows for training: {len(df_combined)}")
        
        processed_path = "/opt/airflow/data/processed_training_data.csv"
        df_combined.to_csv(processed_path, index=False)
        return processed_path

    @task
    def train_challenger_model(processed_path: str):
        df = pd.read_csv(processed_path)
        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]
        
        # Holdout split so both models are evaluated fairly
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        challenger = RandomForestRegressor(n_estimators=50, random_state=42)
        challenger.fit(X_train, y_train)
        
        challenger_path = "/opt/airflow/models/challenger_model.pkl"
        test_data_path = "/opt/airflow/data/holdout_test.csv"
        
        os.makedirs(os.path.dirname(challenger_path), exist_ok=True)
        joblib.dump(challenger, challenger_path)
        
        test_df = X_test.copy()
        test_df["MedHouseVal"] = y_test
        test_df.to_csv(test_data_path, index=False)
        
        return {"challenger_path": challenger_path, "test_data_path": test_data_path}

    @task
    def evaluate_and_deploy(paths: dict):
        prod_path = "/opt/airflow/models/house_model.pkl"
        challenger_path = paths["challenger_path"]
        test_df = pd.read_csv(paths["test_data_path"])
        
        X_test = test_df.drop(columns=["MedHouseVal"])
        y_test = test_df["MedHouseVal"]
        
        # Evaluate Challenger
        challenger = joblib.load(challenger_path)
        challenger_preds = challenger.predict(X_test)
        challenger_r2 = r2_score(y_test, challenger_preds)
        challenger_rmse = root_mean_squared_error(y_test, challenger_preds)
        
        print(f"Challenger Metrics -> R2: {challenger_r2:.4f} | RMSE: {challenger_rmse:.4f}")
        
        # Evaluate Current Production Model (if one exists)
        prod_r2 = -1.0
        if os.path.exists(prod_path):
            prod_model = joblib.load(prod_path)
            prod_preds = prod_model.predict(X_test)
            prod_r2 = r2_score(y_test, prod_preds)
            prod_rmse = root_mean_squared_error(y_test, prod_preds)
            print(f"Production Metrics -> R2: {prod_r2:.4f} | RMSE: {prod_rmse:.4f}")
        
        # Decision Gate
        decision = "REJECTED"
        if challenger_r2 >= MIN_R2_THRESHOLD and challenger_r2 >= prod_r2:
            shutil.move(challenger_path, prod_path)
            decision = "PROMOTED"
            print(f"PROMOTED: Challenger model passed threshold ({MIN_R2_THRESHOLD}) and beat/matched production!")
        else:
            if os.path.exists(challenger_path):
                os.remove(challenger_path)
            print(f"REJECTED: Keeping existing production model. (Challenger R2={challenger_r2:.4f}, Prod R2={prod_r2:.4f}, Threshold={MIN_R2_THRESHOLD})")

        # Save evaluation summary for the UI
        import json
        metrics_summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "min_r2_threshold": MIN_R2_THRESHOLD,
            "challenger_r2": round(float(challenger_r2), 4),
            "challenger_rmse": round(float(challenger_rmse), 4),
            "production_r2": round(float(prod_r2), 4) if prod_r2 != -1.0 else None,
            "production_rmse": round(float(prod_rmse), 4) if prod_r2 != -1.0 else None,
            "decision": decision
        }
        with open("/opt/airflow/models/latest_eval_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

    processed_file = merge_and_clean_data()
    artifacts = train_challenger_model(processed_file)
    evaluate_and_deploy(artifacts)

house_dag = house_retraining_pipeline()
