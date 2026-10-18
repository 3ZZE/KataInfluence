@echo off
rem Influence GUI with the CPU (Eigen) engine, for machines without a working OpenCL GPU. Extra arguments go to the GUI.
cd /d "%~dp0"
influence-gui.exe --engine "./katago-cpu.exe gtp -model model.bin.gz -config gtp.cfg -override-config maxVisits=500" %*
