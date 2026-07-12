import joblib
import pandas as pd
model = joblib.load("predictive_maintenance_model.pkl")

print("========================================")
print(" Predictive Maintenance Prediction")
print("========================================\n")
machine_type = int(input("Machine Type (L=1, M=2, H=0): "))
air_temp = float(input("Air Temperature (K): "))
process_temp = float(input("Process Temperature (K): "))
rotational_speed = int(input("Rotational Speed (rpm): "))
torque = float(input("Torque (Nm): "))
tool_wear = int(input("Tool Wear (minutes): "))
input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
})
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)
print("\nPrediction Probability")
print(f"Normal Machine : {probability[0][0]*100:.2f}%")
print(f"Machine Failure: {probability[0][1]*100:.2f}%")

if prediction[0] == 0:
    print("\nPrediction : NORMAL MACHINE")
else:
    print("\nPrediction : MACHINE FAILURE")
