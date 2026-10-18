Influence - GUI and KataGo-based engine
=======================================

Rules: RULES.md

Start
-----
  Windows: play-gpu.bat   (or play-cpu.bat)
  Linux:   ./play-gpu.sh  (or ./play-cpu.sh)

play-gpu uses the OpenCL engine (katago-opencl), which needs a GPU with OpenCL drivers
(on Linux also the OpenCL ICD loader, e.g. the ocl-icd / ocl-icd-libopencl1 package).
The first start tunes OpenCL for your GPU, which can take a minute or two.

play-cpu uses the CPU engine (katago-cpu, needs a CPU with AVX2). It is much slower, so it
searches fewer visits per move (maxVisits in the play-cpu launcher).

Starting influence-gui on its own opens the board without an engine.
Usage: influence-gui [position.txt] [--engine "<engine command line>"] [--replay <file.sgf|file.sgfs>] [--game <n>]

Controls
--------
Click an own cell to select an attack source, click a neighbour to attack, Space ends the attack
phase, then click own cells to spend reinforcement points (Shift+click spends as many as fit).
Z / Y (or Left / Right) undo and redo, A toggles analysis, E toggles edit mode.
In the engine panel choose who plays Blue and Red (Human or Engine).

Files
-----
  model.bin.gz   neural net (@MODEL@)
  gtp.cfg        engine settings (strength: maxVisits; threads: numSearchThreads)
  gtp_logs/      engine logs, created on first run

The engine is a fork of KataGo (https://github.com/lightvector/KataGo), see LICENSE-KataGo.txt.
