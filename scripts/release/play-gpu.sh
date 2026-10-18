#!/bin/sh
# Influence GUI with the GPU (OpenCL) engine. Extra arguments go to the GUI.
cd "$(dirname "$0")" || exit 1
exec ./influence-gui --engine "./katago-opencl gtp -model model.bin.gz -config gtp.cfg" "$@"
