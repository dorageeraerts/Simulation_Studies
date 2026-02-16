#!/bin/bash
set -e

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
    export GEANT4_DATA_DIR=/root/geant4/data
    echo "[INFO] Geant4 environment loaded. G4INSTALL=$G4INSTALL"
    echo "Working with Geant4 version: $(geant4-config --version)"
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

# -----------------------------
# Launch interactive shell
# -----------------------------
<<<<<<< Updated upstream:docker-singularity/entrypoint.sh
echo "[INFO] Container ready. Tools available in /opt"
exec bash -l


=======

echo "Current working directory: $PWD"
exec bash -l
>>>>>>> Stashed changes:Container/entrypoint.sh
