## AI-Powered Triage Management System

## Overview
This project is an AI-powered triage management system that helps classify patients based on their symptoms and vital signs. It predicts the severity of a patient's condition and assigns them to the appropriate medical department. Additionally, it provides first-aid guidance for relevant symptoms.

## Features
- Predicts patient severity using a trained RandomForest model.
- Assigns patients to the most relevant medical department.
- Provides first-aid guidance based on selected symptoms.
- Stores patient data in a CSV file (`patient_records.csv`).
- User-friendly Tkinter-based GUI for easy data entry and prediction.

## Technologies Used
- Python
- Tkinter (for GUI)
- Pandas (for data handling)
- NumPy (for numerical processing)
- Scikit-Learn (for AI model training)

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas scikit-learn tk
  ```

## Installation & Setup
1. Clone the repository or download the script.
2. Place the `triage_data.csv` file in the same directory as the script.
3. Run the Python script:
   ```bash
   python triage_system.py
   ```

## Usage
1. Enter patient details (Name, Age, Weight, Heart Rate, Blood Pressure, Temperature).
2. Select applicable symptoms from the list.
3. Click the **Predict** button to classify the patient's severity and assign a department.
4. The prediction results, along with first-aid guidance, will be displayed in the same window.
5. Click **Reset** to clear all fields for a new patient entry.

## Data Processing
- The system uses a dataset (`triage_data.csv`) with patient records.
- Categorical labels (e.g., severity and department) are encoded using `LabelEncoder`.
- The input features are standardized using `StandardScaler` before prediction.

## Model Training
- The system uses two `RandomForestClassifier` models:
  1. One for predicting severity.
  2. Another for assigning a medical department.
- The models are trained on preprocessed data and used for real-time predictions.

## File Structure
```
├── triage_system.py   # Main application script
├── triage_data.csv    # Dataset (must be present in the same directory)
├── patient_records.csv # Generated file storing patient entries (created dynamically)
```

## Future Enhancements
- Improve the ML model using deep learning (e.g., Neural Networks).
- Integrate a database instead of CSV for patient records.
- Develop a web-based interface using FastAPI or Django.
- Implement real-time sensor data integration for vitals monitoring.


