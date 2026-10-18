#!/bin/bash
# Runs inside the influence-release container (see make_release.sh).
# /src = repo (read-only), /work = build scratch, /out = where the archives go.
set -euo pipefail
JOBS=${JOBS:-$(nproc)}
MODEL=${MODEL:?}
mkdir -p /work /out

# ---------------- Windows dependencies: zlib and an OpenCL.dll import library ----------------
D=/work/win-deps
if [ ! -f $D/lib/libz.a ]; then
  mkdir -p $D/lib $D/include /work/dl
  cd /work/dl
  curl -sSfL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz | tar xz
  make -C zlib-1.3.1 -f win32/Makefile.gcc PREFIX=x86_64-w64-mingw32- libz.a -j$JOBS >/dev/null
  cp zlib-1.3.1/libz.a $D/lib/
  cp zlib-1.3.1/zlib.h zlib-1.3.1/zconf.h $D/include/
fi
if [ ! -s $D/lib/libOpenCL.a ]; then
  # Windows users get OpenCL.dll from their GPU driver; we only need an import library for it.
  cp -r /usr/include/CL $D/include/
  { echo "LIBRARY OpenCL.dll"; echo "EXPORTS";
    nm -D --defined-only /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 | awk '$2=="T" && $3 ~ /^cl/ {sub(/@.*/, "", $3); print $3}'; } > $D/OpenCL.def
  cd $D && x86_64-w64-mingw32-dlltool -d $D/OpenCL.def -D OpenCL.dll -l $D/lib/libOpenCL.a
fi

# ---------------- Engine ----------------
build_engine() { # os backend
  local os=$1 backend=$2 b=/work/build-$1-$2
  local args=(-S /src/cpp -B $b -DCMAKE_BUILD_TYPE=Release -DUSE_BACKEND=${backend^^})
  [ "$backend" = eigen ] && args+=(-DUSE_AVX2=1)
  if [ "$os" = linux ]; then
    args+=(-DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.a
           -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc")
  else
    args+=(-DCMAKE_TOOLCHAIN_FILE=/src/scripts/release/mingw.cmake
           -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0601 -DNOMINMAX -DBYTE_ORDER=1234 -DLITTLE_ENDIAN=1234 -DBIG_ENDIAN=4321"
           -DCMAKE_CXX_STANDARD_LIBRARIES="-lws2_32 -lshlwapi"
           -DCMAKE_EXE_LINKER_FLAGS="-static -Wl,--stack,8388608"
           -DZLIB_INCLUDE_DIR=$D/include -DZLIB_LIBRARY=$D/lib/libz.a
           -DOpenCL_INCLUDE_DIR=$D/include -DOpenCL_LIBRARY=$D/lib/libOpenCL.a
           -DEIGEN3_INCLUDE_DIRS=/usr/include/eigen3)
  fi
  cmake "${args[@]}" >/dev/null
  cmake --build $b -j$JOBS 2>&1 | grep -E -i 'error|warning: unused|undefined' || true
  local exe=katago; [ "$os" = windows ] && exe=katago.exe
  test -f $b/$exe
}
for os in linux windows; do for backend in opencl eigen; do
  echo "== engine $os $backend"; build_engine $os $backend
done; done

# ---------------- GUI ----------------
export CARGO_TARGET_DIR=/work/gui-target
cd /src/gui
echo "== gui linux";   cargo build --release --locked -q
echo "== gui windows"; cargo build --release --locked -q --target x86_64-pc-windows-gnu

# ---------------- Package ----------------
package() { # os
  local os=$1 name=influence-$1-x64 x=""; [ "$os" = windows ] && x=.exe
  local p=/work/pkg/$name
  rm -rf $p; mkdir -p $p
  cp /work/build-$os-opencl/katago$x $p/katago-opencl$x
  cp /work/build-$os-eigen/katago$x  $p/katago-cpu$x
  if [ "$os" = linux ]; then
    cp $CARGO_TARGET_DIR/release/influence-gui $p/
    strip $p/katago-opencl $p/katago-cpu $p/influence-gui
  else
    cp $CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/influence-gui.exe $p/
    x86_64-w64-mingw32-strip $p/*.exe
  fi
  cp /src/models/run1/$MODEL $p/model.bin.gz
  cp /src/gui/gtp.cfg $p/gtp.cfg
  cp /src/RULES.md $p/
  cp /src/LICENSE $p/LICENSE-KataGo.txt
  cp /src/scripts/release/README.txt $p/
  sed -i "s/@MODEL@/$MODEL/" $p/README.txt
  if [ "$os" = linux ]; then
    cp /src/scripts/release/play-gpu.sh /src/scripts/release/play-cpu.sh $p/
    chmod +x $p/*.sh $p/katago-* $p/influence-gui
    tar -C /work/pkg -czf /out/$name.tar.gz $name
  else
    for f in play-gpu play-cpu; do sed 's/$/\r/' /src/scripts/release/$f.bat > $p/$f.bat; done
    sed -i 's/$/\r/' $p/README.txt
    (cd /work/pkg && rm -f /out/$name.zip && zip -qr /out/$name.zip $name)
  fi
}
package linux
package windows
ls -l /out
