#!/bin/bash
set -e
chmod +x *.sh

source muraves_mulder_config.sh

./generate_bins.sh

./submit_all_bins.sh