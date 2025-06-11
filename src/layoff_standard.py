import pandas as pd
import numpy as np

# load the original dataset
file_path = "layoffs_2025.csv"
df = pd.read_csv(file_path)

# clean the '%' column and convert to float
df['% Clean'] = df['%'].str.replace('%', '', regex=False).astype(float)

# create the unified layoffs_value column
df['layoffs_value'] = np.where(df['% Clean'].notna(), df['% Clean'], df['# Laid off'])

# create the is_percent indicator column
df['is_percent'] = np.where(df['% Clean'].notna(), 1, 0)

# df = df.drop(columns=['% Clean'])

#save to a new CSV
df.to_csv("layoffs_standardized.csv", index=False)
