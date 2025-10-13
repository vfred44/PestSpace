# %%
import json

# %%

input_file = '/Users/fredvaartnou/VSCODE/PestSpace/PestSpace monitoring - 2025-09-29 09_16_28.477020+00_00.json'
output_file = "/Users/fredvaartnou/Desktop/PestSpace_extracted.json"

# Load full JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# Iterate over all sampling areas
for area in data.get("sampling_areas", []):
    for event in area.get("sampling_events", []):
        for sample in event.get("material_samples", []):
            
            # interactions → interacting_object.name
            interacting_object_name = ""
            for interaction in sample.get("interactions", []):
                interacting_obj = interaction.get("interacting_object", {})
                if isinstance(interacting_obj, dict):
                    interacting_object_name = interacting_obj.get("name", "")
                    break  # take only first
            
            # material_sample.name
            name = sample.get("name", "")
            
            # measurements → first value if available
            measurements = sample.get("measurements", [])
            values = [m.get("value", "") for m in measurements if isinstance(m, dict)]
            value = values[0] if values else ""
            
            # files → all download_links
            files = sample.get("files", [])
            download_links = [f.get("download_link", "") for f in files if isinstance(f, dict)]
            
            results.append({
                "interacting_object_name": interacting_object_name,
                "name": name,
                "value": value,
                "download_links": download_links
            })

# Save extracted results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"✅ Extracted {len(results)} samples")
print(f"Results saved to {output_file}")




# %%