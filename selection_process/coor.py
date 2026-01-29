import pandas as pd
df = pd.read_excel("synthetic_equalweights_quiz50pct.xlsx")

df[[col for col in df.columns if col != 'label']].corrwith(df['label']).sort_values(ascending=False)
