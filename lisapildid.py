# %%
import pandas as pd
import os
import requests

# %%
df = pd.read_csv('/Users/fredvaartnou/Desktop/Taimekahjustused ML/PestSpace failid/PestSpace-extracted_lisapildid.csv', sep=",")

# %%
df.loc[df[".value"] == "Septoria tritici blotch", ".interacting_object_name"] = "Triticum aestivum"
df.loc[df[".value"] == "Tan spot", ".interacting_object_name"] = "Triticum aestivum"
df.loc[df[".value"] == "Stagonospora blotch", ".interacting_object_name"] = "Triticum aestivum"
df.loc[df[".value"] == "Chocolate spot", ".interacting_object_name"] = "Vicia faba"
df.loc[df[".value"] == "Downy mildew", ".interacting_object_name"] = "Vicia faba"
df.loc[df[".value"] == "Alternaria blight", ".interacting_object_name"] = "Vicia faba"
df.loc[df[".value"] == "Brown rust", ".interacting_object_name"] = "Triticum aestivum"
df.loc[df[".value"] == "Pink snow mould", ".interacting_object_name"] = "Triticum aestivum"

# %%
df.columns = ["Taim", "Kood", "Haigus", "Pilt"]
df.to_csv("/Users/fredvaartnou/Desktop/Taimekahjustused ML/PestSpace failid/lisapildid.csv", index=False)

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


