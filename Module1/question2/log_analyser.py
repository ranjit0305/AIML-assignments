import pandas as pd
records = []
with open("application.log", "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        level = parts[2]
        module = parts[3]
        if level in ["ERROR", "WARNING"]:
            records.append({
                "Level": level,
                "Module": module
            })
# Convert to DataFrame
df = pd.DataFrame(records)
# Count frequency by module and level
report = (
    df.groupby(["Module", "Level"])
      .size()
      .reset_index(name="Count")
)
print(report)
# Save report
report.to_csv("error_report.csv", index=False)
print("\nCSV report generated successfully!")