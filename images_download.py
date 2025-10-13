# %%
import pandas as pd
import os
import requests

# %%
df = pd.read_csv('/Users/fredvaartnou/Desktop/Taimekahjustused ML/PestSpace failid/PestSpace_filtered_lisa_8.10.2025.csv', sep=";")

# %%
df = df.replace(" ", "_", regex = True)

# %%
os.makedirs("images", exist_ok=True)

for pilt, kood, taim, haigus in zip(df["Pilt"], df["Kood"], df["Taim"], df["Haigus"]):
    filename = f"{kood}_{taim}_{haigus}.jpg"
    save_path = os.path.join("images", filename)

    counter = 2
    while os.path.exists(save_path):
        filename = f"{kood}_{taim}_{haigus}_{counter}.jpg"
        save_path = os.path.join("images", filename)
        counter +=1

    with open(save_path, "wb") as f:
        f.write(requests.get(pilt).content)

# %%



