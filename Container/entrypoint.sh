#!/bin/bash
set -e

# -----------------------------
# Set paths for simulation tools
# -----------------------------
# EcoMug headers
export CPLUS_INCLUDE_PATH=/opt/EcoMug:$CPLUS_INCLUDE_PATH

# CRY, PUMAS, Turtle binaries
export PATH=/opt/CRY/bin:/opt/PUMAS/bin:/opt/Turtle/examples:$PATH

# Optional: Turtle library path if needed
export LD_LIBRARY_PATH=/opt/Turtle/lib:$LD_LIBRARY_PATH

# -----------------------------
# Welcome message
# -----------------------------
echo " ~~~~~~~  Welcome to the Muraves Geant4 Simulation Container 0.0 ~~~~~~~~  "
echo "                      .-----. "
echo "             .----. .'       ' "
echo "            '      V           '  "
echo "          '                      ' "
echo "        '                          '   "
echo "      '                              ' "
echo "       _  _        _   _        _  _  "
echo "      |  V | |  | |_| |_| \\  / |_ |_  "
echo "      |    | |__| | \\ | |  \\/  |_  _| "
echo
echo " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  "
echo "Tools installed in /opt:"
ls /opt


# -----------------------------
# Source environment for Geant4 via geant4-config
# -----------------------------
if command -v geant4-config >/dev/null 2>&1; then
    export G4INSTALL=$(geant4-config --prefix)
    export PATH=$G4INSTALL/bin:$PATH
    export LD_LIBRARY_PATH=$(geant4-config --libdir):$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$(geant4-config --includedir):$CPLUS_INCLUDE_PATH
    echo "[INFO] Geant4 environment loaded. G4INSTALL=$G4INSTALL"
    #echo "[DEBUG] G4INSTALL=$G4INSTALL"
    #echo "[DEBUG] PATH=$PATH"
    #echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    #echo "[DEBUG] CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH"
else
    echo "[WARNING] geant4-config not found. Geant4 environment not loaded."
fi

#------------------------------------------------
# Check if /workspace is empty or not a git repo
#------------------------------------------------
if [ ! -d "/workspace/.git" ]; then
    echo "[WARNING:] No Git repo found in /workspace. "
else
    echo "Detected existing Git repo in /workspace."
fi

echo "[INFO] Container ready."
# -----------------------------
# Launch interactive shell
# -----------------------------

# -----------------------------
# Container identity & custom prompt
# -----------------------------
export CONTAINER_NAME="muraves-sim"
export CLUSTER_USER="${USER}"   # host username
export PS1="[\[\e[36m\]${CONTAINER_NAME}\[\e[0m\] | \[\e[32m\]${CLUSTER_USER}\[\e[0m\] | \[\e[34m\]\w\[\e[0m\]]\$ "

exec bash -l