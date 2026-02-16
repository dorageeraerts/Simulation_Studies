#!/bin/bash
set -e
# -----------------------------
# Wrapper script to run Simulation-Studies container on T2B
# -----------------------------
# This file is NOT needed to build the image.
# It is used to:
# - run the container
# - automatically start the environment via entrypoint
# - check existence of Git repository in the binded workspace

echo "=== Setting up Simulation-Studies container ==="

# -----------------------------
# Ask user for workspace location
# -----------------------------
DEFAULT_WORKSPACE="$HOME/MURAVES"
read -rp "Enter workspace location on host [default: $DEFAULT_WORKSPACE]: " WORKSPACE_INPUT

if [ -z "$WORKSPACE_INPUT" ]; then
    WORKSPACE_HOST="$DEFAULT_WORKSPACE"
else
    WORKSPACE_HOST="$(eval echo "$WORKSPACE_INPUT")"
fi

export WORKSPACE_HOST
echo "[INFO] Using workspace: $WORKSPACE_HOST"

# Create workspace if it doesn't exist
mkdir -p "$WORKSPACE_HOST"

# -----------------------------
# Clone repository if not present
# -----------------------------
REPO_URL="https://github.com/muraves/Simulation_Studies.git"
WORKSPACE_REPO="$WORKSPACE_HOST/Simulation_Studies"

if [ ! -d "$WORKSPACE_REPO/.git" ]; then
    echo "[INFO] Cloning Simulation_Studies repository..."
    git clone "$REPO_URL" "$WORKSPACE_REPO"
else
    echo "[INFO] Repository already exists: $WORKSPACE_REPO"
fi

# Move into repo (important if script is run from elsewhere)
cd "$WORKSPACE_REPO"

# Ensure entrypoint is executable
#chmod +x environment/docker/entrypoint.sh

# -----------------------------
# Paths to Singularity image and bindings
# -----------------------------
#SIF_PATH="/user/dgeeraer/muraves-sim.sif"
#SIF_PATH="$HOME/MURAVES/Simulation_Studies/Container/muraves-sim.sif"
SIF_PATH="/group/Muography/MURAVES/container/simulation_container/muraves-sim-latest.sif"

ENTRYPOINT_HOST="/group/Muography/MURAVES/container/simulation_container/entrypoint.sh"
ENTRYPOINT_CONTAINER="/workspace/Simulation_Studies/entrypoint.sh"

WORKSPACE_CONTAINER="/workspace"

#DIR_HOST="/pnfs/iihe/muraves/muraves_DATA"
DIR_HOST="$HOME"
DIR_CONTAINER="/data"

#OUTPUT_HOST="/group/Muography/MURAVES/"
OUTPUT_HOST="$HOME"
OUTPUT_CONTAINER="/outputs"

G4DATA_HOST="/group/Muography/MURAVES/container/simulation_container/geant4_datasets"
G4DATA_CONTAINER="/root/geant4/data"

# -----------------------------
# Default command if none provided
# -----------------------------
if [ $# -eq 0 ]; then
    CMD="bash"
else
    CMD="$@"
fi

# -----------------------------
# Test bind: ensure entrypoint is visible
# -----------------------------
#echo "[INFO] Testing bind for entrypoint inside container..."
#singularity exec \
    #--bind ${WORKSPACE_HOST}:${WORKSPACE_CONTAINER},${DIR_HOST}:${DIR_CONTAINER},${OUTPUT_HOST}:${OUTPUT_CONTAINER},${ENTRYPOINT_HOST}:${ENTRYPOINT_CONTAINER} \
    #${SIF_PATH} \
    #ls -l ${ENTRYPOINT_CONTAINER}

# -----------------------------
# Execute entrypoint inside the container
# -----------------------------
echo "[INFO] Launching container with entrypoint..."
singularity exec \
    --bind ${WORKSPACE_HOST}:${WORKSPACE_CONTAINER},\
${DIR_HOST}:${DIR_CONTAINER},\
${OUTPUT_HOST}:${OUTPUT_CONTAINER},\
${ENTRYPOINT_HOST}:${ENTRYPOINT_CONTAINER},\
${G4DATA_HOST}:${G4DATA_CONTAINER} \
    --pwd ${WORKSPACE_CONTAINER}/Simulation_Studies \
    ${SIF_PATH} \
    ${ENTRYPOINT_CONTAINER} ${CMD}
    #bash -l -c "source ${ENTRYPOINT_CONTAINER} && exec bash"
    #/workspace/Simulation_Studies/path/to/entrypoint.sh ${CMD}
    #${G4DATA_HOST}:${G4DATA_CONTAINER} \ 
