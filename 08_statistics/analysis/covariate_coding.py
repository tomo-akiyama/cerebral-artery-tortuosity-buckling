# Covariate coding
import pandas as pd
import os

# csv file path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

# Load csv file
df = pd.read_csv(csv_path)

# --- Recode ethnicity ---
ethnicity_map = {
    1: 'white',
    2: 'unknown',
    3: 'asian',
    4: 'black',
    5: 'asian',
    6: 'other'
}
df['ethnicity'] = df['ETHNIC_ID'].map(ethnicity_map)

print("\nCount by Ethnicity:")
print(df['ethnicity'].value_counts(dropna=False).sort_index())

# Recode ethnicity into a simplified group (for modeling)
ethnicity_group_map = {
    'black': 'black/other/unknown',
    'other': 'black/other/unknown',
    'unknown': 'black/other/unknown',
    'white': 'white',
    'asian': 'asian'
}

df['ethnicity_group'] = df['ethnicity'].map(ethnicity_group_map)
print("\nCount by Ethnicity_group:")
print(df['ethnicity_group'].value_counts(dropna=False).sort_index())

# --- Recode sex ---
# 1 = male, 2 = female
sex_map = {1: 'male', 2: 'female'}
df['sex'] = df['SEX_ID (1=m, 2=f)'].map(sex_map)

print(f"Number of males: {(df['sex'] == 'male').sum()}")
print(f"Number of females: {(df['sex'] == 'female').sum()}")

print("================================================")
print("QC:")
print(f"Missing ethnicity labels: {df['ethnicity'].isna().sum()}")
print(f"Missing ethnicity_group labels: {df['ethnicity_group'].isna().sum()}")
print(f"Missing sex labels: {df['sex'].isna().sum()}")

# Save processed csv
df.to_csv(csv_path, index=False)