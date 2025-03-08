import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load and preprocess data
data = pd.read_csv(r"D:\Coding\ai-triage-system\triage_data.csv")
data = data.drop(columns=["id", "name"], errors='ignore')

severity_encoder = LabelEncoder()
data["severity"] = severity_encoder.fit_transform(data["severity"])

def assign_department(row):
    if row.get("chief_complaint_chest_pain", 0) or row.get("chief_complaint_difficulty_breathing", 0):
        return "Cardiology"
    elif row.get("chief_complaint_abdominal_pain", 0) or row.get("chief_complaint_nausea", 0):
        return "Gastroenterology"
    elif row.get("chief_complaint_fracture", 0) or row.get("chief_complaint_burn_injury", 0):
        return "Orthopedics"
    elif row.get("chief_complaint_stroke_symptoms", 0) or row.get("chief_complaint_unconscious", 0):
        return "Neurology"
    else:
        return "General Medicine"

data["department"] = data.apply(assign_department, axis=1)

department_encoder = LabelEncoder()
data["department"] = department_encoder.fit_transform(data["department"])

X = data.drop(columns=["severity", "department"], errors='ignore')
X = X.apply(pd.to_numeric, errors='coerce')
y_severity = data["severity"]
y_department = data["department"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
model_severity = RandomForestClassifier()
model_department = RandomForestClassifier()

model_severity.fit(X_scaled, y_severity)
model_department.fit(X_scaled, y_department)

# Define symptom categories
exclude_features = {"age", "weight", "heart_rate", "blood_pressure", "temperature"}
symptoms = [col for col in X.columns if col not in exclude_features]

first_aid_guidance = {
    "chief_complaint_chest_pain": "Chew and swallow an aspirin while waiting for emergency services.",
    "chief_complaint_difficulty_breathing": "Sit in an upright position and take slow deep breaths.",
    "chief_complaint_abdominal_pain": "Apply a warm compress and avoid eating heavy meals.",
    "chief_complaint_nausea": "Sip clear fluids and avoid strong odors.",
    "chief_complaint_fracture": "Immobilize the injured area and apply ice packs.",
    "chief_complaint_burn_injury": "Cool the burn under running water for at least 10 minutes.",
    "chief_complaint_stroke_symptoms": "Lay the person down with their head elevated and call emergency services.",
    "chief_complaint_unconscious": "Check for breathing and place in recovery position if necessary."
}

def get_first_aid(input_data):
    aid_list = [first_aid_guidance[symptom] for i, symptom in enumerate(symptoms) if input_data[i] == 1 and symptom in first_aid_guidance]
    return "\n".join(aid_list) if aid_list else "No specific first aid required."

def save_patient_data(name, age, weight, heart_rate, blood_pressure, temperature, severity, department):
    filename = "patient_records.csv"
    new_data = pd.DataFrame([[name, age, weight, heart_rate, blood_pressure, temperature, severity, department]],
                             columns=["Name", "Age", "Weight", "Heart Rate", "Blood Pressure", "Temperature", "Severity", "Department"])
    if os.path.exists(filename):
        new_data.to_csv(filename, mode='a', header=False, index=False)
    else:
        new_data.to_csv(filename, mode='w', header=True, index=False)

def predict():
    try:
        name = entry_name.get()
        age = float(entry_age.get())
        weight = float(entry_weight.get())
        heart_rate = float(entry_heart_rate.get())
        blood_pressure = float(entry_blood_pressure.get())
        temperature = float(entry_temperature.get())

        input_data = [age, weight, heart_rate, blood_pressure, temperature] + [int(var.get()) for var in symptom_vars]
        input_data = np.array(input_data, dtype=np.float64).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        severity_pred = model_severity.predict(input_data_scaled)[0]
        department_pred = model_department.predict(input_data_scaled)[0]
        severity_text = severity_encoder.inverse_transform([severity_pred])[0]
        department_text = department_encoder.inverse_transform([department_pred])[0]
        first_aid = get_first_aid(input_data[0])

        save_patient_data(name, age, weight, heart_rate, blood_pressure, temperature, severity_text, department_text)
        
        label_result.config(text=f"Predicted Severity: {severity_text}\nAssigned Department: {department_text}\nFirst Aid Guidance: {first_aid}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Triage System")
root.geometry("600x700")
root.configure(bg="#f0f0f0")

frame_main = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
frame_main.pack()

fields = ["Name", "Age", "Weight", "Heart Rate", "Blood Pressure", "Temperature"]
entries = {}

for i, label in enumerate(fields):
    tk.Label(frame_main, text=label + ":", font=("Arial", 10, "bold"), bg="#f0f0f0").grid(row=i, column=0, sticky="w", pady=2)
    entry = tk.Entry(frame_main, width=30)
    entry.grid(row=i, column=1, pady=2)
    entries[label.lower()] = entry

entry_name = entries["name"]
entry_age = entries["age"]
entry_weight = entries["weight"]
entry_heart_rate = entries["heart rate"]
entry_blood_pressure = entries["blood pressure"]
entry_temperature = entries["temperature"]

symptom_vars = []
tk.Label(frame_main, text="Symptoms:", font=("Arial", 10, "bold"), bg="#f0f0f0").grid(row=len(fields), column=0, sticky="w", pady=5)
frame_symptoms = tk.Frame(frame_main, bg="#ffffff")
frame_symptoms.grid(row=len(fields) + 1, column=0, columnspan=2, pady=5)

for symptom in symptoms:
    var = tk.IntVar()
    tk.Checkbutton(frame_symptoms, text=symptom.replace("chief_complaint_", "").replace("_", " ").title(), variable=var, bg="#ffffff").pack(anchor="w")
    symptom_vars.append(var)

tk.Button(frame_main, text="Predict", command=predict, bg="#4CAF50", fg="white").grid(pady=10)
label_result = tk.Label(frame_main, text="", font=("Arial", 10), bg="#f0f0f0", justify="left", wraplength=500)
label_result.grid(pady=10)

root.mainloop()
