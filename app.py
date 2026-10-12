from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import os
import threading
import traceback

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"

DATA = None
MODEL = None
MODEL_STATUS = {"ready": False, "training": False, "message": "Model has not been trained yet."}
MODEL_LOCK = threading.Lock()

REQUIRED_COLUMNS = ["Dates", "Category", "DayOfWeek", "PdDistrict", "X", "Y"]

def load_data():
    global DATA
    if DATA is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"train.csv not found at {DATA_PATH}. Copy train.csv beside app.py."
            )
        df = pd.read_csv(DATA_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError("Dataset is missing columns: " + ", ".join(missing))
        df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
        df = df.dropna(subset=["Dates", "Category", "DayOfWeek", "PdDistrict"]).copy()
        df["Hour"] = df["Dates"].dt.hour
        df["Month"] = df["Dates"].dt.month
        df["Year"] = df["Dates"].dt.year
        DATA = df
    return DATA

def get_features(df):
    # Keep the model's inputs simple, interpretable, and available from user selections.
    return df[["Hour", "Month", "DayOfWeek", "PdDistrict"]].astype(str)

def train_model():
    global MODEL, MODEL_STATUS
    with MODEL_LOCK:
        if MODEL_STATUS["training"]:
            return
        MODEL_STATUS = {"ready": False, "training": True, "message": "Training model on a representative sample..."}
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib

        df = load_data()
        # Sampling makes first-time training practical on a student laptop.
        sample_n = min(80000, len(df))
        sample = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df.copy()
        X = get_features(sample)
        y = sample["Category"].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        cat_features = ["Hour", "Month", "DayOfWeek", "PdDistrict"]
        preprocessor = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), cat_features)],
            remainder="drop"
        )
        model = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=60, max_depth=18, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ])
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        MODEL = model
        joblib.dump(model, BASE_DIR / "crime_model.joblib")
        MODEL_STATUS = {
            "ready": True,
            "training": False,
            "message": f"Model trained on {len(X_train):,} records; holdout accuracy {accuracy:.1%}.",
            "accuracy": round(float(accuracy), 4),
            "trained_rows": int(len(X_train)),
            "test_rows": int(len(X_test))
        }
    except Exception as exc:
        MODEL_STATUS = {"ready": False, "training": False, "message": f"Model training failed: {exc}"}

def ensure_model():
    global MODEL, MODEL_STATUS
    if MODEL is not None:
        return
    model_path = BASE_DIR / "crime_model.joblib"
    if model_path.exists():
        try:
            import joblib
            MODEL = joblib.load(model_path)
            MODEL_STATUS = {"ready": True, "training": False, "message": "Loaded saved model."}
            return
        except Exception:
            pass
    if not MODEL_STATUS["training"]:
        thread = threading.Thread(target=train_model, daemon=True)
        thread.start()

@app.route("/")
def home():
    try:
        load_data()
        return render_template("index.html")
    except Exception as exc:
        return render_template("setup_error.html", error=str(exc)), 500

@app.get("/api/summary")
def api_summary():
    try:
        df = load_data()
        total = int(len(df))
        top = df["Category"].value_counts().head(10)
        districts = sorted(df["PdDistrict"].dropna().astype(str).unique().tolist())
        categories = sorted(df["Category"].dropna().astype(str).unique().tolist())
        years = sorted(df["Year"].dropna().astype(int).unique().tolist())
        return jsonify({
            "total": total,
            "category_count": int(df["Category"].nunique()),
            "district_count": int(df["PdDistrict"].nunique()),
            "top_category": str(top.index[0]) if len(top) else "N/A",
            "top_category_count": int(top.iloc[0]) if len(top) else 0,
            "districts": districts,
            "categories": categories,
            "years": years,
            "top_categories": [{"category": str(k), "count": int(v)} for k, v in top.items()],
            "days": [{"day": str(k), "count": int(v)} for k, v in df["DayOfWeek"].value_counts().items()],
            "hours": [{"hour": int(k), "count": int(v)} for k, v in df.groupby("Hour").size().items()],
            "months": [{"month": int(k), "count": int(v)} for k, v in df.groupby("Month").size().items()],
            "district_counts": [{"district": str(k), "count": int(v)} for k, v in df["PdDistrict"].value_counts().items()],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.get("/api/analytics")
def api_analytics():
    try:
        df = load_data()
        district = request.args.get("district", "All")
        category = request.args.get("category", "All")
        year_from = request.args.get("year_from", type=int)
        year_to = request.args.get("year_to", type=int)
        if district != "All":
            df = df[df["PdDistrict"].astype(str) == district]
        if category != "All":
            df = df[df["Category"].astype(str) == category]
        if year_from is not None:
            df = df[df["Year"] >= year_from]
        if year_to is not None:
            df = df[df["Year"] <= year_to]
        if df.empty:
            return jsonify({"total": 0, "top_categories": [], "days": [], "hours": [], "months": [], "district_counts": []})
        return jsonify({
            "total": int(len(df)),
            "top_categories": [{"category": str(k), "count": int(v)} for k, v in df["Category"].value_counts().head(10).items()],
            "days": [{"day": str(k), "count": int(v)} for k, v in df["DayOfWeek"].value_counts().items()],
            "hours": [{"hour": int(k), "count": int(v)} for k, v in df.groupby("Hour").size().items()],
            "months": [{"month": int(k), "count": int(v)} for k, v in df.groupby("Month").size().items()],
            "district_counts": [{"district": str(k), "count": int(v)} for k, v in df["PdDistrict"].value_counts().items()],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.post("/api/train")
def api_train():
    ensure_model()
    return jsonify(MODEL_STATUS)

@app.get("/api/model-status")
def api_model_status():
    ensure_model()
    return jsonify(MODEL_STATUS)

@app.post("/api/predict")
def api_predict():
    try:
        ensure_model()
        if MODEL is None:
            return jsonify({"error": MODEL_STATUS.get("message", "Model is still training. Please try again shortly."), "status": MODEL_STATUS}), 503
        payload = request.get_json(force=True)
        district = str(payload.get("district", "")).strip()
        day = str(payload.get("day", "")).strip()
        hour = int(payload.get("hour", 12))
        month = int(payload.get("month", 6))
        df = load_data()
        if district not in set(df["PdDistrict"].astype(str).unique()):
            return jsonify({"error": "Please choose a valid police district."}), 400
        if day not in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}:
            return jsonify({"error": "Please choose a valid day."}), 400
        if not 0 <= hour <= 23 or not 1 <= month <= 12:
            return jsonify({"error": "Hour must be 0–23 and month must be 1–12."}), 400
        features = pd.DataFrame([{
            "Hour": str(hour), "Month": str(month),
            "DayOfWeek": day, "PdDistrict": district
        }])
        probabilities = MODEL.predict_proba(features)[0]
        classes = MODEL.named_steps["classifier"].classes_
        top_indices = np.argsort(probabilities)[::-1][:5]
        predictions = [{"category": str(classes[i]), "confidence": round(float(probabilities[i]), 4)} for i in top_indices]
        return jsonify({
            "predicted_category": predictions[0]["category"],
            "predictions": predictions,
            "note": "Experimental estimate learned from historical San Francisco reports; it is not a forecast that a crime will occur."
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
