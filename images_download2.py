# %%
import pandas as pd
import os
import requests

# %%
df = pd.read_csv('/Users/fredvaartnou/Desktop/Taimekahjustused ML/PestSpace failid/PestSpace_filtered_lisa_8.10.2025.csv', sep=";")

# %%
df = df.replace(" ", "_", regex = True)

# %%
for pilt, kood, taim, haigus in zip(df["Pilt"], df["Kood"], df["Taim"], df["Haigus"]):

    # Create subfolder for the plant and disease
    base_dir = os.path.join("images", taim, haigus)
    os.makedirs(base_dir, exist_ok=True)

    # Start with a base filename
    counter = 1
    while True:
        if counter == 1:
            filename = f"{kood}_{taim}_{haigus}.jpg"
        else:
            filename = f"{kood}_{taim}_{haigus}_{counter}.jpg"

        save_path = os.path.join(base_dir, filename)

        # Break loop if filename does not exist
        if not os.path.exists(save_path):
            break

        counter += 1  # Increment to try next filename

    # Download and save the image
    try:
        response = requests.get(pilt)
        response.raise_for_status()  # raise error if request fails
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download {pilt}: {e}")




# %%
