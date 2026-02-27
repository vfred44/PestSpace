#!/bin/bash

# === CONFIGURE THESE ===
LOCAL_DIR="/Users/fredvaartnou/VSCODE/PestSpace/"              # local folder
HPC_USER="fred87"                       # your HPC username
HPC_HOST="rocket.hpc.ut.ee"		# HPC hostname
HPC_DIR="/gpfs/helios/home/fred87/PestSpace"       # destination on HPC

# === IGNORE LIST ===
EXCLUDES=(
    "--exclude=.gitignore"
    "--exclude=PestSpace/"    
    "--exclude=lightning_logs/"
    "--exclude=Notebooks/"
    "--exclude=outputs/"
    "--exclude=UTHPC/"
    "--exclude=wandb/"
    "--exclude=sync_to_hpc.sh"
    "--exclude=environment.yml"
)

# Build the exclude arguments
EXCLUDE_ARGS=""
for e in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS $e"
done

# === RSYNC COMMAND ===
rsync -avz --progress \
    --include="conf/" --include="conf/***" \
    --include="data/" --include="data/***" \
    --include="models/" --include="models/***" \
    --include="run_model.sh" \
    --include="train.py" \
    --include="debugging.py" \
    --exclude="*" \
    $EXCLUDE_ARGS \
    "$LOCAL_DIR" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}"
