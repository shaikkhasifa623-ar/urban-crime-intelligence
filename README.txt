URBAN CRIME INTELLIGENCE — FULL-STACK PROJECT

This project separates the frontend from the backend:
- Frontend: HTML + CSS + JavaScript + Plotly charts
- Backend: Python Flask REST API
- ML: scikit-learn Random Forest classifier
- Dataset: your existing San Francisco `train.csv`

SETUP
1. Extract this ZIP to a normal folder.
2. Copy your `train.csv` file into the same folder as `app.py`.
3. Open the extracted folder in VS Code (File > Open Folder).
4. Open Terminal > New Terminal.
5. Install packages:
   python -m pip install -r requirements.txt
6. Start the backend and website:
   python app.py
7. Open this address in Chrome:
   http://127.0.0.1:5000

HOW TO USE
- Overview cards and charts load from Flask API endpoints.
- Use filters and press Apply filters to update charts from the backend.
- In the ML lab, choose district, day, hour and month, then click Generate estimate.
- The first prediction may take a minute because the Random Forest model is trained on a representative sample and saved as crime_model.joblib.
- Later runs load the saved model.

IMPORTANT
- This is an educational prototype. It models the reported crime category conditional on historical records; it does not predict whether a crime will occur.
- Data is historical San Francisco data, not Hyderabad data.
- Model confidence is not a calibrated real-world probability.
- If training fails, read the model-status message in the dashboard/terminal.
- Do not upload train.csv publicly if it contains data you are not permitted to redistribute.
