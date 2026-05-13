import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("waqi786/mental-health-and-technology-usage-dataset")

print("Path to dataset files:", path)

# List files in the path
files = os.listdir(path)
print("Files:", files)

# If there is a csv file, load it and show info
for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, file))
        print(f"\nInfo for {file}:")
        print(df.info())
        print(df.head())
        break
