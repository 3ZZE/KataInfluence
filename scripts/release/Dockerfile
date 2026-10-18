# Build environment for release archives: Linux binaries against Ubuntu 22.04's glibc (2.35),
# Windows binaries cross-compiled with MinGW-w64 (posix threads).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl ca-certificates xz-utils zip file pkg-config \
      zlib1g-dev ocl-icd-opencl-dev opencl-headers libeigen3-dev \
      g++-mingw-w64-x86-64-posix gcc-mingw-w64-x86-64-posix binutils-mingw-w64-x86-64 \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
      --target x86_64-pc-windows-gnu
ENV CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc-posix
RUN git config --system --add safe.directory '*'
