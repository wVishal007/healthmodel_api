import pandas as pd

# Read CSV file
data = pd.read_csv("symptoms.csv")   # <-- replace with your csv filename

# Select only first 5 values (rows)
subset = data.head(5)

# Save to text file
with open("output.txt", "w") as f:
    f.write(subset.to_string(index=False))
