@echo off
rem Influence GUI with the GPU (OpenCL) engine. Extra arguments go to the GUI.
cd /d "%~dp0"
influence-gui.exe --engine "./katago-opencl.exe gtp -model model.bin.gz -config gtp.cfg" %*
