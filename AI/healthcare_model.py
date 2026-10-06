import math

MEDICAL_DISCLAIMER = "This is an AI-assisted screening tool and not a substitute for a qualified healthcare professional."

def validate_vitals(age, heart_rate, spo2, temp_c, bp_sys, bp_dia, resp_rate, glucose, bmi):
    """
    Validates physical boundaries for vitals. Returns (is_valid, warning_message)
    """
    warnings = []
    if age is not None and (age < 0 or age > 120):
        warnings.append("Age out of normal human range (0-120).")
    if heart_rate is not None and (heart_rate < 30 or heart_rate > 220):
        warnings.append("Heart rate out of physiological range (30-220 bpm).")
    if spo2 is not None and (spo2 < 50 or spo2 > 100):
        warnings.append("SpO2 percentage out of range (50-100%).")
    if temp_c is not None and (temp_c < 30.0 or temp_c > 45.0):
        warnings.append("Body temperature out of physiological range (30-45°C).")
    if bp_sys is not None and (bp_sys < 50 or bp_sys > 260):
        warnings.append("Systolic BP out of range (50-260 mmHg).")
    if bp_dia is not None and (bp_dia < 30 or bp_dia > 160):
        warnings.append("Diastolic BP out of range (30-160 mmHg).")
    if glucose is not None and (glucose < 30 or glucose > 600):
        warnings.append("Blood glucose out of range (30-600 mg/dL).")
    if bmi is not None and (bmi < 10.0 or bmi > 70.0):
        warnings.append("BMI out of range (10-70 kg/m²).")

    is_valid = len(warnings) == 0
    return is_valid, "; ".join(warnings) if warnings else "Vitals within physiological ranges."

def evaluate_screening_risk(data):
    """
    Evaluates clinical risk score and triage level based on patient vitals, symptoms, and features.
    
    Data input keys:
    - age (int/float)
    - sex (str: 'M', 'F', 'Other')
    - heart_rate (float)
    - spo2 (float)
    - temperature (float)
    - bp_sys (float)
    - bp_dia (float)
    - respiratory_rate (float)
    - glucose (float)
    - bmi (float)
    - symptoms (str)
    - medical_history (str)
    """
    age = data.get("age", 35)
    hr = data.get("heart_rate")
    spo2 = data.get("spo2")
    temp = data.get("temperature")
    bp_sys = data.get("bp_sys") or data.get("systolic_bp")
    bp_dia = data.get("bp_dia") or data.get("diastolic_bp")
    resp = data.get("respiratory_rate")
    glucose = data.get("glucose")
    bmi = data.get("bmi")
    symptoms = str(data.get("symptoms", "")).lower()

    risk_score = 0.0
    risk_factors = []

    # 1. SpO2 Hypoxia Check (Critical Safety)
    if spo2 is not None:
        if spo2 < 90.0:
            risk_score += 4.0
            risk_factors.append(f"Severe Hypoxia (SpO2 {spo2:.1f}%)")
        elif spo2 < 94.0:
            risk_score += 2.0
            risk_factors.append(f"Moderate Hypoxia (SpO2 {spo2:.1f}%)")

    # 2. Blood Pressure Check
    if bp_sys is not None or bp_dia is not None:
        sys_val = bp_sys or 120
        dia_val = bp_dia or 80
        if sys_val >= 180 or dia_val >= 120:
            risk_score += 4.0
            risk_factors.append(f"Hypertensive Crisis ({sys_val:.0f}/{dia_val:.0f} mmHg)")
        elif sys_val >= 140 or dia_val >= 90:
            risk_score += 2.0
            risk_factors.append(f"Stage 2 Hypertension ({sys_val:.0f}/{dia_val:.0f} mmHg)")
        elif sys_val >= 130 or dia_val >= 80:
            risk_score += 1.0
            risk_factors.append(f"Stage 1 Hypertension ({sys_val:.0f}/{dia_val:.0f} mmHg)")

    # 3. Blood Glucose Check
    if glucose is not None:
        if glucose >= 250.0:
            risk_score += 3.5
            risk_factors.append(f"Severe Hyperglycemia (Glucose {glucose:.0f} mg/dL)")
        elif glucose >= 180.0:
            risk_score += 2.0
            risk_factors.append(f"High Glucose (Glucose {glucose:.0f} mg/dL)")
        elif glucose < 70.0:
            risk_score += 2.5
            risk_factors.append(f"Hypoglycemia Warning (Glucose {glucose:.0f} mg/dL)")

    # 4. Heart Rate Check
    if hr is not None:
        if hr > 130 or hr < 40:
            risk_score += 3.0
            risk_factors.append(f"Abnormal Heart Rate ({hr:.0f} bpm)")
        elif hr > 100 or hr < 50:
            risk_score += 1.0
            risk_factors.append(f"Tachycardia/Bradycardia ({hr:.0f} bpm)")

    # 5. Temperature Check
    if temp is not None:
        if temp >= 39.5 or temp <= 35.0:
            risk_score += 2.5
            risk_factors.append(f"High Fever / Hypothermia ({temp:.1f}°C)")
        elif temp >= 38.0:
            risk_score += 1.0
            risk_factors.append(f"Fever ({temp:.1f}°C)")

    # 6. Critical Symptom Keywords
    critical_keywords = ["chest pain", "shortness of breath", "difficulty breathing", "unconscious", "stroke", "severe dizziness", "bleeding"]
    moderate_keywords = ["fever", "cough", "fatigue", "headache", "nausea", "numbness"]

    for kw in critical_keywords:
        if kw in symptoms:
            risk_score += 3.0
            risk_factors.append(f"Critical Symptom Identified: '{kw}'")
            break

    for kw in moderate_keywords:
        if kw in symptoms:
            risk_score += 1.0
            risk_factors.append(f"Symptom Reported: '{kw}'")
            break

    # Categorize Risk Level & Next Action
    if risk_score >= 5.0:
        risk_level = "CRITICAL"
        rec_action = "Urgent medical evaluation and immediate emergency escalation recommended."
        confidence = 0.94
    elif risk_score >= 3.0:
        risk_level = "HIGH"
        rec_action = "Prompt clinical evaluation by a qualified healthcare worker recommended."
        confidence = 0.89
    elif risk_score >= 1.5:
        risk_level = "MODERATE"
        rec_action = "Schedule follow-up health screening and monitor vitals closely."
        confidence = 0.85
    else:
        risk_level = "LOW"
        rec_action = "Vitals and screening parameters are within routine limits. Continue standard care."
        confidence = 0.92

    if not risk_factors:
        risk_factors.append("No adverse physiological factors detected.")

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "confidence": confidence,
        "key_factors": risk_factors,
        "recommended_action": rec_action,
        "disclaimer": MEDICAL_DISCLAIMER
    }
