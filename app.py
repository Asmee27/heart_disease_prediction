# app.py

from flask import Flask, render_template, request, jsonify, send_file, make_response, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-random-secret'  # needed for session

# --- Utilities ---
def json_safe(value):
    """Recursively convert numpy types to native Python for JSON/session."""
    if isinstance(value, dict):
        return {json_safe(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(json_safe(v) for v in value)
    try:
        import numpy as _np
        if isinstance(value, (_np.integer,)):
            return int(value)
        if isinstance(value, (_np.floating,)):
            return float(value)
        if isinstance(value, (_np.bool_,)):
            return bool(value)
        if isinstance(value, (_np.ndarray,)):
            return json_safe(value.tolist())
    except Exception:
        pass
    return value

# --- Configuration & Model Loading ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_assets():
    """Loads all necessary machine learning assets (6 .pkl files)."""
    try:
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'heart_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        log_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_model.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        cluster_centers = joblib.load(os.path.join(MODEL_DIR, 'cluster_centers.pkl'))
        association_rules = joblib.load(os.path.join(MODEL_DIR, 'association_rules.pkl'))
        
        return rf_model, scaler, log_model, feature_names, cluster_centers, association_rules
    except Exception as e:
        print(f"Error loading assets: {e}")
        # Return placeholders for the 6 assets if loading fails
        return None, None, None, None, None, None

RF_MODEL, SCALER, LOG_MODEL, FEATURE_NAMES, CLUSTER_CENTERS, ASSOCIATION_RULES = load_assets()

# --- Feature Mapping (Must align with train_model.py's encoding) ---
MAPPING = {
    'Gender': {'Male': 1, 'Female': 0},
    'Exercise Habits': {'High': 0, 'Medium': 1, 'Low': 2},
    'Smoking': {'Yes': 1, 'No': 0},
    'Family Heart Disease': {'Yes': 1, 'No': 0},
    'Diabetes': {'Yes': 1, 'No': 0},
    'High Blood Pressure': {'Yes': 1, 'No': 0},
    'Low HDL Cholesterol': {'Yes': 1, 'No': 0},
    'High LDL Cholesterol': {'Yes': 1, 'No': 0},
    'Alcohol Consumption': {'High': 0, 'Medium': 1, 'Low': 2, 'None': 3},
    'Stress Level': {'High': 0, 'Medium': 1, 'Low': 2},
}

CONTINUOUS_FEATURES = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
    'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level', 'Sugar Consumption'
]

# --- Helper Functions ---

def get_prediction_insights(rf_model, form_data):
    """Processes form data, makes prediction, and generates detailed text output with full risk advice for all user-specified logic."""
    # 1. Prepare data frame from form inputs
    input_data = {}
    for feature in FEATURE_NAMES:
        value = form_data.get(feature)
        if feature in MAPPING:
            input_data[feature] = MAPPING[feature].get(value, 0)
        else:
            try:
                input_data[feature] = float(value)
            except (ValueError, TypeError):
                input_data[feature] = 0.0
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    # 2. Make prediction
    prediction_int = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0]
    prob_yes = probability[1]
    # 3. Generate Detailed Insights (Analysis & Reasons)
    reasons = []
    user_risks = []
    #--- FULL user-specified logic/advice mapping ---#
    if input_df['Cholesterol Level'].iloc[0] > 240:
        reasons.append("Your Cholesterol Level is significantly elevated (> 240 mg/dL). Adopt a low-fat, high-fiber diet. Limit red meat/fried foods. Consider medical consultation for statins.")
        user_risks.append({
            'emoji':'🩸', 'title':'High Cholesterol', 'color':'red',
            'tip':"Let’s cut down fried foods and enjoy more fiber-rich meals!"})
    if 'High LDL Cholesterol' in input_df.columns and input_df['High LDL Cholesterol'].iloc[0] == 1:
        reasons.append("Bad cholesterol (LDL) is elevated. Include more oats, nuts, omega-3 rich foods. Avoid trans fats.")
        user_risks.append({
            'emoji':'💔', 'title':'High LDL Cholesterol', 'color':'red',
            'tip':"Bad cholesterol (LDL) is elevated, More oats, nuts, and healthy fats will help lower LDL!"})
    if 'Low HDL Cholesterol' in input_df.columns and input_df['Low HDL Cholesterol'].iloc[0] == 1:
        reasons.append("Protective cholesterol (HDL) is low. Increase aerobic activity and healthy fats like olive oil and fish.")
        user_risks.append({
            'emoji':'🥑', 'title':'Low HDL Cholesterol', 'color':'yellow',
            'tip':"Let’s get moving—try regular walks and a bit more olive oil or fish!"})
    if input_df['Triglyceride Level'].iloc[0] > 200:
        reasons.append("Your Triglyceride Level is high (> 200 mg/dL). Limit sugar and alcohol intake. Prefer complex carbs like brown rice.")
        user_risks.append({
            'emoji':'🍚', 'title':'High Triglycerides', 'color':'orange',
            'tip':"Let’s switch to brown rice and cut back on sodas or sweets!"})
    if 'Smoking' in input_df.columns and input_df['Smoking'].iloc[0] == MAPPING['Smoking']['Yes']:
        reasons.append("Smoking detected as a major modifiable risk. Quit smoking; risk reduces by 50% within one year. Consider nicotine replacement therapy.")
        user_risks.append({
            'emoji':'🚭', 'title':'Smoking', 'color':'red',
            'tip':"Smoking detected. Quitting is the single best step for your heart!"})
    if 'Alcohol Consumption' in input_df.columns and (form_data.get('Alcohol Consumption', '').lower() == 'high' or input_df['Alcohol Consumption'].iloc[0] == MAPPING['Alcohol Consumption']['High']):
        reasons.append("Heavy alcohol consumption. Limit alcohol to ≤1 drink/day (women) or ≤2 (men). Abstain if liver or BP issues present.")
        user_risks.append({
            'emoji':'🍷', 'title':'Alcohol Use', 'color':'orange',
            'tip':"Heavy alcohol consumption. Let’s swap that extra drink for a fun activity or mocktail!"})
    if input_df['Blood Pressure'].iloc[0] > 140:
        reasons.append("Blood Pressure > 140/90 mmHg detected. Hypertension increases heart strain. Reduce salt intake, exercise, monitor BP weekly.")
        user_risks.append({
            'emoji':'💓', 'title':'High Blood Pressure', 'color':'red',
            'tip':"Keep checking your BP, move daily, and sprinkle less salt!"})
    if 'Family Heart Disease' in input_df.columns and input_df['Family Heart Disease'].iloc[0] == 1:
        reasons.append("Family history of heart disease. Genetic predisposition increases baseline risk. Annual checkups, lipid profile, ECG monitoring recommended.")
        user_risks.append({
            'emoji':'❤️', 'title':'Family Risk', 'color':'yellow',
            'tip':"Annual checkups and staying proactive helps you stay ahead!"})
    if prediction_int == 1:
        result_text = "Heart Disease **PRESENT**"
        conclusion = "The model suggests your health profile aligns with a high-risk group. It is imperative to consult a physician immediately."
        if not reasons:
            reasons = ["Multiple low-to-medium risk factors are cumulatively contributing to this elevated risk."]
    else:
        result_text = "Heart Disease NOT PRESENT"
        conclusion = "Your health metrics align with a lower-risk profile. Continue to maintain a healthy lifestyle and monitor key indicators."
        if not reasons:
            reasons = ["Your current favorable health metrics (e.g., cholesterol, blood pressure) contribute to this low-risk assessment."]
    return {
        'prediction': result_text,
        'probability': f"{prob_yes * 100:.2f}%",
        'conclusion': conclusion,
        'reasons': reasons,
        'is_risk': prediction_int == 1,
        'user_risks': user_risks
    }

def get_cluster_plot_data(input_data, scaler, cluster_centers, feature_names):
    """Returns (user_x, user_y, centroids:list of dicts for 2D visualization)"""
    # We'll use the first two features (in feature_names) as axes
    # Cluster centers: rows = cluster, columns = features
    # Scale input like was done for clustering
    if scaler is None or cluster_centers is None:
        return None, None, []
    num_features = list(cluster_centers.columns[:2])  # x & y axes
    # Create DataFrame with full feature set for scaler consistency
    df_row = pd.DataFrame([[input_data.get(f, 0) for f in cluster_centers.columns]], columns=cluster_centers.columns)
    X_array = scaler.transform(df_row)
    user_x = float(X_array[0][0])
    user_y = float(X_array[0][1])
    # Collect clusters/centroids
    centroids = []
    for i, row in cluster_centers.iterrows():
        centroids.append({
            'x': float(row[num_features[0]]),
            'y': float(row[num_features[1]]),
            'idx': int(i)
        })
    return user_x, user_y, centroids, num_features


# --- Routes ---

@app.route('/')
def home():
    """Renders the Home Page."""
    return render_template('home.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    """Renders the Predictor Page and handles POST requests for prediction."""
    
    # Structure features for the HTML form display
    feature_groups = {
        'Personal & Demographic': ['Age', 'Gender', 'BMI'],
        'Cardiovascular Metrics': ['Blood Pressure', 'Cholesterol Level', 'Triglyceride Level', 'Fasting Blood Sugar'],
        'Lifestyle & Habits': ['Exercise Habits', 'Smoking', 'Alcohol Consumption', 'Sleep Hours', 'Stress Level'],
        'Existing Conditions': ['Family Heart Disease', 'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol'],
        'Inflammatory Markers': ['CRP Level', 'Homocysteine Level', 'Sugar Consumption']
    }
    
    prediction_output = None
    cluster_plot_data = None
    user_coords = None
    cluster_axes = None
    user_health_data = None
    healthy_ref_values = None
    user_risks = None
    user_metric_values = None
    healthy_metric_benchmarks = None
    if request.method == 'POST':
        if RF_MODEL is None:
            prediction_output = {'prediction': "Model Error", 'error': "Machine learning assets not loaded. Run train_model.py first."}
        else:
            prediction_output = get_prediction_insights(RF_MODEL, request.form)
            # --- Prepare cluster plot data for visualization ---
            input_data = {}
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)
                if feature in MAPPING:
                    input_data[feature] = MAPPING[feature].get(value, 0)
                else:
                    try:
                        input_data[feature] = float(value)
                    except (ValueError, TypeError):
                        input_data[feature] = 0.0
            user_x, user_y, clusters, axes = get_cluster_plot_data(input_data, SCALER, CLUSTER_CENTERS, FEATURE_NAMES)
            user_coords = {'x': user_x, 'y': user_y}
            cluster_plot_data = clusters
            cluster_axes = axes
            # -- Prepare fixed metric comparison (requested metrics) --
            user_metric_values = {
                'Blood Pressure': input_data.get('Blood Pressure', 0),
                'Cholesterol Level': input_data.get('Cholesterol Level', 0),
                'Fasting Blood Sugar': input_data.get('Fasting Blood Sugar', 0),
                'CRP Level': input_data.get('CRP Level', 0),
                'Triglyceride Level': input_data.get('Triglyceride Level', 0),
                'BMI': input_data.get('BMI', 0)
            }
            healthy_metric_benchmarks = {
                'Blood Pressure': 120,
                'Cholesterol Level': 200,
                'Fasting Blood Sugar': 90,
                'CRP Level': 1.0,
                'Triglyceride Level': 150,
                'BMI': 22
            }
            # Map to template variables expected by predictor.html
            user_health_data = {
                'Blood Pressure': user_metric_values['Blood Pressure'],
                'Cholesterol Level': user_metric_values['Cholesterol Level'],
                'Fasting Blood Sugar': user_metric_values['Fasting Blood Sugar'],
                'CRP Level': user_metric_values['CRP Level'],
                'Triglyceride Level': user_metric_values['Triglyceride Level'],
                'BMI': user_metric_values['BMI'],
            }
            healthy_ref_values = {
                'Blood Pressure': healthy_metric_benchmarks['Blood Pressure'],
                'Cholesterol Level': healthy_metric_benchmarks['Cholesterol Level'],
                'Fasting Blood Sugar': healthy_metric_benchmarks['Fasting Blood Sugar'],
                'CRP Level': healthy_metric_benchmarks['CRP Level'],
                'Triglyceride Level': healthy_metric_benchmarks['Triglyceride Level'],
                'BMI': healthy_metric_benchmarks['BMI'],
            }
            # Personalized risks for cards
            user_risks = prediction_output.get('user_risks', [])
            # Save needed data in session for report
            session['report_input'] = json_safe(input_data)
            session['report_output'] = json_safe(prediction_output)
            session['report_user_risks'] = json_safe(user_risks)
            session['report_user_metrics'] = json_safe(user_metric_values)
            session['report_healthy_metrics'] = json_safe(healthy_metric_benchmarks)
            session['feature_names'] = json_safe(list(FEATURE_NAMES))
    return render_template('predictor.html', 
                           feature_groups=feature_groups,
                           mapping=MAPPING, 
                           output=prediction_output,
                           cluster_plot_data=cluster_plot_data,
                           user_coords=user_coords,
                           cluster_axes=cluster_axes,
                           user_health_data=user_health_data,
                           healthy_ref_values=healthy_ref_values,
                           user_metric_values=user_metric_values,
                           healthy_metric_benchmarks=healthy_metric_benchmarks,
                           user_risks=user_risks)

def _status_emoji(val, good_low=None, good_high=None, mod_low=None, mod_high=None):
    # Returns tuple (emoji, label_color)
    if good_low is None and good_high is None:
        return ('🟡', 'yellow')
    if val is None:
        return ('🟡', 'yellow')
    # Determine ranges; risky if outside moderate
    if val < good_low:
        return ('🟡', 'yellow')
    if good_high is not None and val <= good_high:
        return ('🟢', 'green')
    if mod_high is not None and val <= mod_high:
        return ('🟡', 'yellow')
    return ('🔴', 'red')

@app.route('/download_report')
def download_report():
    # Require prior prediction data
    report_input = session.get('report_input')
    report_output = session.get('report_output')
    user_risks = session.get('report_user_risks', [])
    user_metrics = session.get('report_user_metrics')
    healthy_metrics = session.get('report_healthy_metrics')
    if not report_input or not report_output:
        return redirect(url_for('predictor'))
    # Build User Input Summary rows
    def rng(low, high):
        return f"{low}–{high}" if low is not None and high is not None else "—"
    rows = []
    age = report_input.get('Age')
    bmi = report_input.get('BMI')
    bp = report_input.get('Blood Pressure')
    chol = report_input.get('Cholesterol Level')
    sugar = report_input.get('Sugar Consumption')
    crp = report_input.get('CRP Level')
    # Define healthy ranges
    # Age: informational only
    rows.append({'metric':'Age','value':age,'range':'—','status':('🟡','yellow')})
    # BMI
    bmi_emoji, bmi_color = _status_emoji(bmi, good_low=18.5, good_high=24.9, mod_low=25, mod_high=29.9)
    rows.append({'metric':'BMI','value':bmi,'range':'18.5–24.9','status':(bmi_emoji,bmi_color)})
    # Blood Pressure (systolic)
    bp_emoji, bp_color = _status_emoji(bp, good_low=0, good_high=120, mod_low=121, mod_high=139)
    rows.append({'metric':'Blood Pressure','value':bp,'range':'<120 (systolic)','status':(bp_emoji,bp_color)})
    # Cholesterol
    chol_emoji, chol_color = _status_emoji(chol, good_low=0, good_high=200, mod_low=201, mod_high=239)
    rows.append({'metric':'Cholesterol','value':chol,'range':'<200 mg/dL','status':(chol_emoji,chol_color)})
    # Sugar Consumption (g/day)
    sug_emoji, sug_color = _status_emoji(sugar, good_low=0, good_high=25, mod_low=26, mod_high=50)
    rows.append({'metric':'Sugar Consumption','value':sugar,'range':'≤25 g/day','status':(sug_emoji,sug_color)})
    # CRP Level
    crp_emoji, crp_color = _status_emoji(crp, good_low=0, good_high=1, mod_low=1.01, mod_high=3)
    rows.append({'metric':'CRP Level','value':crp,'range':'<1.0 mg/L','status':(crp_emoji,crp_color)})

    # Top influencing factors: take top 5 from RF feature importances
    top_factors = []
    try:
        importances = list(RF_MODEL.feature_importances_)
        names = list(FEATURE_NAMES)
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:5]
        top_factors = [n for n, _ in ranked]
    except Exception:
        top_factors = []

    return render_template('report.html',
                           rows=rows,
                           output=report_output,
                           user_risks=user_risks,
                           user_metrics=user_metrics,
                           healthy_metrics=healthy_metrics,
                           top_factors=top_factors)

@app.route('/analysis')
def analysis():
    """Analysis Page: Displays feature importance, model comparison, Clustering, and Pattern Mining."""
    if RF_MODEL is None or LOG_MODEL is None or CLUSTER_CENTERS is None or ASSOCIATION_RULES is None:
        return render_template('analysis.html', error="One or more model/analysis assets not loaded. Run train_model.py first.")

    # --- Feature Importance Data (from RF Model) ---
    feature_importances_raw = RF_MODEL.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': feature_importances_raw
    }).sort_values(by='Importance', ascending=False).head(10)
    
    # --- Model Comparison Data ---
    # Load data for metric calculation
    df = pd.read_csv(os.path.join(MODEL_DIR, 'heart_disease.csv'))
    
    # Handle target variable mapping first, before other encoding
    df['Heart Disease Status'] = df['Heart Disease Status'].map({'No': 0, 'Yes': 1})
    
    # Simple imputation/encoding (mimic train_model.py for metrics calculation)
    for col in df.columns:
        if col == 'Heart Disease Status':
            continue  # Skip target variable as it's already encoded
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            if col in MAPPING:
                 df[col] = df[col].map(MAPPING.get(col, {}))
            else: # Encode any remaining object columns
                 df[col] = df[col].astype('category').cat.codes
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    X = df.drop("Heart Disease Status", axis=1)
    y = df["Heart Disease Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare test data for Logistic Regression
    # Scale all features as the scaler was trained on all features
    X_test_log = X_test.copy()
    X_test_scaled_array = SCALER.transform(X_test_log)

    # Assign the scaled array back to the DataFrame
    X_test_log = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)

    rf_accuracy = accuracy_score(y_test, RF_MODEL.predict(X_test))
    log_accuracy = accuracy_score(y_test, LOG_MODEL.predict(X_test_log))
    
    comparison_data = [
        {'model': 'Random Forest Classifier', 'accuracy': rf_accuracy, 'type': 'Ensemble'},
        {'model': 'Logistic Regression', 'accuracy': log_accuracy, 'type': 'Linear'}
    ]

    # --- Clustering Data ---
    cluster_data = CLUSTER_CENTERS.T.reset_index()
    cluster_data.columns = ['Feature', 'Cluster 0 (Low-to-Medium Risk)', 'Cluster 1 (High-Risk Profile)', 'Cluster 2 (Medium-Risk Profile)']
    
    # --- Pattern Mining Data ---
    rules_data = ASSOCIATION_RULES[['antecedents', 'consequents', 'lift']].head(5).to_dict('records')


    return render_template('analysis.html', 
                           importance_data=feature_importance_df.to_dict('records'), 
                           comparison_data=comparison_data,
                           cluster_data=cluster_data.to_dict('records'),
                           rules_data=rules_data)


if __name__ == '__main__':
    # Running with debug=False and use_reloader=False ensures stable process execution
    app.run(debug=False, use_reloader=False)