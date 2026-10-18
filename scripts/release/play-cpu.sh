#!/bin/sh
# Influence GUI with the CPU (Eigen) engine, for machines without a working OpenCL GPU. Extra arguments go to the GUI.
cd "$(dirname "$0")" || exit 1
exec ./influence-gui --engine "./katago-cpu gtp -model model.bin.gz -config gtp.cfg -override-config maxVisits=500" "$@"
