import wandb
import json

api = wandb.Api()

run = api.run("fred-vaartnou-ut/PestSpace/f1iscw83")

data = {
    "config": dict(run.config),
    "summary": run.summary._json_dict
}

with open("wandb_run_overview.json", "w") as f:
    json.dump(data, f, indent=4)

print("Export complete")