#!/bin/bash

CLUSTER_ID=$1
PROC_ID=$2

SIF_PATH="/group/Muography/MURAVES/container/simulation_container/muraves-sim-latest.sif"
G4DATA_HOST="/group/Muography/MURAVES/container/simulation_container/geant4_datasets"
G4DATA_CONTAINER="/root/geant4/datasets"

NAME="MARIAELENA"
DETECTOR="NERO"

OUTPUT_TMP=$TMPDIR/${CLUSTER_ID}_${PROC_ID}
#OUTPUT_HOST="/user/dgeeraer/MURAVES/output_simulation/${CLUSTER_ID}"
OUTPUT_HOST="/user/dgeeraer/MURAVES/test_output_sim/Hsphere_placement"
mkdir -p "${OUTPUT_HOST}"
OUTPUT_CONTAINER="/output"
#OUTPUT_PNFS="/pnfs/iihe/muraves/muraves_SIM/FS_UR"

#NAME_FLAG="test_pathlength"
NAME_FLAG="${NAME}_${DETECTOR}"

mkdir -p "${OUTPUT_TMP}"

singularity exec \
    --bind ${G4DATA_HOST}:${G4DATA_CONTAINER},\
${OUTPUT_TMP}:${OUTPUT_CONTAINER} \
    --pwd /user/dgeeraer/MURAVES/Simulation_Studies/Geant4-simulation/build-t2b \
    --env GEANT4_DATA_DIR=${G4DATA_CONTAINER},CLUSTER_ID=${CLUSTER_ID},PROCESS_ID=${PROC_ID},NAME_FLAG=${NAME_FLAG} \
    ${SIF_PATH} \
    ./MuravesSim --m run.mac

echo "Copying output to host..."

for src_file in "${OUTPUT_TMP}"/*; do
    [ -f "$src_file" ] || continue
    filename=$(basename "$src_file")
    dest_file="${OUTPUT_HOST}/${filename}"
    tmp_dest="${dest_file}.tmp_${CLUSTER_ID}_${PROC_ID}"

    # Copy to a hidden temporary name first, then atomically rename
    rsync -av "${src_file}" "${tmp_dest}" \
        && mv "${tmp_dest}" "${dest_file}" \
        || { echo "ERROR: failed to copy ${filename}"; exit 1; }

    echo "  -> ${filename} done"
done

#echo "Done"