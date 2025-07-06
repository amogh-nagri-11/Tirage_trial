import tkinter as tk
from tkinter import messagebox, ttk
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
    "chief_complaint_chest_pain": "🫀 Chew and swallow an aspirin while waiting for emergency services.",
    "chief_complaint_difficulty_breathing": "🫁 Sit in an upright position and take slow deep breaths.",
    "chief_complaint_abdominal_pain": "🔥 Apply a warm compress and avoid eating heavy meals.",
    "chief_complaint_nausea": "💧 Sip clear fluids and avoid strong odors.",
    "chief_complaint_fracture": "🧊 Immobilize the injured area and apply ice packs.",
    "chief_complaint_burn_injury": "🚿 Cool the burn under running water for at least 10 minutes.",
    "chief_complaint_stroke_symptoms": "🚨 Lay the person down with their head elevated and call emergency services.",
    "chief_complaint_unconscious": "⚕️ Check for breathing and place in recovery position if necessary."
}

def get_first_aid(input_data):
    aid_list = [first_aid_guidance[symptom] for i, symptom in enumerate(symptoms) 
                if input_data[i] == 1 and symptom in first_aid_guidance]
    return "\n".join(aid_list) if aid_list else "✅ No specific first aid required."

def save_patient_data(name, age, weight, heart_rate, blood_pressure, temperature, severity, department):
    filename = "patient_records.csv"
    new_data = pd.DataFrame([[name, age, weight, heart_rate, blood_pressure, temperature, severity, department]],
                            columns=["Name", "Age", "Weight", "Heart Rate", "Blood Pressure", "Temperature", "Severity", "Department"])
    if os.path.exists(filename):
        new_data.to_csv(filename, mode='a', header=False, index=False)
    else:
        new_data.to_csv(filename, mode='w', header=True, index=False)

def clear_form():
    """Clear all form fields"""
    for entry in entries.values():
        entry.delete(0, tk.END)
    for var in symptom_vars:
        var.set(0)
    label_result.config(text="")

def predict():
    try:
        name = entry_name.get()
        if not name.strip():
            messagebox.showwarning("Warning", "Please enter patient name")
            return
            
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
        
        # Color coding for severity
        severity_colors = {
            "Critical": "#e74c3c",
            "High": "#f39c12",
            "Medium": "#f1c40f",
            "Low": "#27ae60"
        }
        
        result_text = f"🏥 TRIAGE ASSESSMENT COMPLETE\n\n"
        result_text += f"📋 Patient: {name}\n"
        result_text += f"⚡ Severity: {severity_text}\n"
        result_text += f"🏢 Department: {department_text}\n\n"
        result_text += f"🩺 First Aid Guidance:\n{first_aid}"
        
        label_result.config(text=result_text, fg=severity_colors.get(severity_text, "#2c3e50"))
        
        # Show success message
        messagebox.showinfo("Success", f"Patient {name} has been successfully triaged!\nSeverity: {severity_text}\nDepartment: {department_text}")
        
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values for all fields")
    except Exception as e:
        messagebox.showerror("Error", f"Error in prediction: {str(e)}")

# GUI Setup with modern styling
root = tk.Tk()
root.title("🏥 AI-Powered Medical Triage System")
root.geometry("800x900")
root.configure(bg="#ecf0f1")

# Configure modern styles
style = ttk.Style()
style.theme_use('clam')

# Main header
header_frame = tk.Frame(root, bg="#34495e", height=80)
header_frame.pack(fill="x", pady=(0, 20))
header_frame.pack_propagate(False)

header_label = tk.Label(header_frame, text="🏥 AI Medical Triage System", 
                        font=("Arial", 24, "bold"), fg="white", bg="#34495e")
header_label.pack(expand=True)

subtitle_label = tk.Label(header_frame, text="Intelligent Patient Assessment & Department Assignment", 
                            font=("Arial", 10), fg="#bdc3c7", bg="#34495e")
subtitle_label.pack()

# Main container with scrollable frame
main_canvas = tk.Canvas(root, bg="#ecf0f1")
scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
scrollable_frame = tk.Frame(main_canvas, bg="#ecf0f1")

scrollable_frame.bind(
    "<Configure>",
    lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
)

main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
main_canvas.configure(yscrollcommand=scrollbar.set)

main_canvas.pack(side="left", fill="both", expand=True, padx=20)
scrollbar.pack(side="right", fill="y")

# Patient Information Section
patient_frame = tk.LabelFrame(scrollable_frame, text="👤 Patient Information", 
                                font=("Arial", 12, "bold"), fg="#2c3e50", bg="#ffffff",
                                relief="ridge", bd=2, padx=20, pady=15)
patient_frame.pack(fill="x", pady=(0, 20), padx=20)

fields = [
    ("Name", "text", "👤"),
    ("Age", "years", "🎂"),
    ("Weight", "kg", "⚖️"),
    ("Heart Rate", "bpm", "💓"),
    ("Blood Pressure", "mmHg", "🩸"),
    ("Temperature", "°C", "🌡️")
]

entries = {}

for i, (label, unit, icon) in enumerate(fields):
    row = i // 2
    col = (i % 2) * 3
    
    # Icon and label
    icon_label = tk.Label(patient_frame, text=icon, font=("Arial", 14), bg="#ffffff")
    icon_label.grid(row=row, column=col, padx=(0, 5), pady=8, sticky="e")
    
    label_widget = tk.Label(patient_frame, text=f"{label} ({unit}):", 
                            font=("Arial", 10, "bold"), fg="#34495e", bg="#ffffff")
    label_widget.grid(row=row, column=col+1, padx=(0, 10), pady=8, sticky="w")
    
    # Entry field with modern styling
    entry = tk.Entry(patient_frame, width=15, font=("Arial", 10), 
                    relief="solid", bd=1, highlightthickness=1,
                    highlightcolor="#3498db", highlightbackground="#bdc3c7")
    entry.grid(row=row, column=col+2, padx=(0, 20), pady=8, sticky="w")
    entries[label.lower()] = entry

# Map entries to variables
entry_name = entries["name"]
entry_age = entries["age"]
entry_weight = entries["weight"]
entry_heart_rate = entries["heart rate"]
entry_blood_pressure = entries["blood pressure"]
entry_temperature = entries["temperature"]

# Symptoms Section
symptoms_frame = tk.LabelFrame(scrollable_frame, text="🩺 Symptoms & Complaints", 
                                font=("Arial", 12, "bold"), fg="#2c3e50", bg="#ffffff",
                                relief="ridge", bd=2, padx=20, pady=15)
symptoms_frame.pack(fill="x", pady=(0, 20), padx=20)

symptom_vars = []
symptoms_grid = tk.Frame(symptoms_frame, bg="#ffffff")
symptoms_grid.pack(fill="x", pady=10)

# Create symptom checkboxes in a grid layout
for i, symptom in enumerate(symptoms):
    row = i // 2
    col = i % 2
    
    var = tk.IntVar()
    symptom_text = symptom.replace("chief_complaint_", "").replace("_", " ").title()
    
    # Add medical icons for different symptoms
    symptom_icons = {
        "Chest Pain": "🫀", "Difficulty Breathing": "🫁", "Abdominal Pain": "🤕",
        "Nausea": "🤢", "Fracture": "🦴", "Burn Injury": "🔥",
        "Stroke Symptoms": "🧠", "Unconscious": "😵"
    }
    
    icon = symptom_icons.get(symptom_text, "🔸")
    
    checkbox = tk.Checkbutton(symptoms_grid, text=f"{icon} {symptom_text}", 
                                variable=var, font=("Arial", 10), bg="#ffffff",
                                activebackground="#e8f4f8", fg="#2c3e50")
    checkbox.grid(row=row, column=col, sticky="w", padx=20, pady=5)
    symptom_vars.append(var)

# Buttons Section
buttons_frame = tk.Frame(scrollable_frame, bg="#ecf0f1")
buttons_frame.pack(fill="x", pady=20, padx=20)

# Predict Button
predict_btn = tk.Button(buttons_frame, text="🔍 Analyze Patient", command=predict,
                        font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                        relief="flat", padx=30, pady=10, cursor="hand2",
                        activebackground="#219a52")
predict_btn.pack(side="left", padx=(0, 10))

# Clear Button
clear_btn = tk.Button(buttons_frame, text="🗑️ Clear Form", command=clear_form,
                        font=("Arial", 12, "bold"), bg="#e74c3c", fg="white",
                        relief="flat", padx=30, pady=10, cursor="hand2",
                        activebackground="#c0392b")
clear_btn.pack(side="left")

# Results Section
results_frame = tk.LabelFrame(scrollable_frame, text="📊 Triage Results", 
                                font=("Arial", 12, "bold"), fg="#2c3e50", bg="#ffffff",
                                relief="ridge", bd=2, padx=20, pady=15)
results_frame.pack(fill="x", pady=(20, 0), padx=20)

label_result = tk.Label(results_frame, text="📋 Complete the patient assessment above to view triage results", 
                        font=("Arial", 11), bg="#ffffff", fg="#7f8c8d",
                        justify="left", wraplength=650, anchor="w")
label_result.pack(fill="x", pady=15)

# Footer
footer_frame = tk.Frame(scrollable_frame, bg="#ecf0f1", height=50)
footer_frame.pack(fill="x", pady=(20, 0))

footer_label = tk.Label(footer_frame, text="⚕️ Powered by AI • For Medical Professional Use Only", 
                        font=("Arial", 9), fg="#95a5a6", bg="#ecf0f1")
footer_label.pack(expand=True)

# Bind mousewheel to canvas for scrolling
def on_mousewheel(event):
    main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

root.bind("<MouseWheel>", on_mousewheel)

# Add hover effects for buttons
def on_enter(event):
    event.widget.config(relief="raised")

def on_leave(event):
    event.widget.config(relief="flat")

predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)
clear_btn.bind("<Enter>", on_enter)
clear_btn.bind("<Leave>", on_leave)

root.mainloop()