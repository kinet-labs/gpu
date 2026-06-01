# Installation Guide {#install}

Build and installation instructions for Kinet GPU Core.

## Requirements

### Build Requirements

- **CMake** 3.20 or later
- **C++17** compatible compiler:
  - GCC 9+
  - Clang 10+
  - MSVC 2019+
- **Optional**: OpenMP for CPU backend parallelization

### Platform Support

| Platform | Compiler | Status |
|----------|----------|--------|
| macOS arm64 | Apple Clang 14+ | Supported |
| macOS x86_64 | Apple Clang 14+ | Supported |
| Linux x86_64 | GCC 9+ / Clang 10+ | Supported |
| Linux aarch64 | GCC 9+ / Clang 10+ | Supported |
| Windows x64 | MSVC 2019+ | Supported |

---

## Quick Start

### Clone and Build

```bash
git clone https://github.com/kinetfi/gpu.git
cd gpu

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

### Install System-Wide

```bash
# Install to /usr/local (requires sudo on Linux/macOS)
sudo cmake --install build

# Or specify custom prefix
cmake --install build --prefix /opt/kinet
```

---

## Build Options

Configure with CMake options:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DKINET_GPU_BUILD_TESTS=ON \
    -DKINET_GPU_BUILD_BENCHMARKS=OFF \
    -DKINET_GPU_CPU_BACKEND=ON \
    -DKINET_GPU_CPU_USE_OPENMP=ON
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `KINET_GPU_BUILD_TESTS` | ON | Build test suite |
| `KINET_GPU_BUILD_BENCHMARKS` | OFF | Build benchmark harness |
| `KINET_GPU_CPU_BACKEND` | ON | Build CPU fallback backend |
| `KINET_GPU_CPU_USE_OPENMP` | ON | Use OpenMP for CPU parallelization |
| `CMAKE_BUILD_TYPE` | - | Debug, Release, RelWithDebInfo |

---

## Platform-Specific Instructions

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# For Apple Silicon (M1/M2/M3), Metal backend is recommended
# Install separately from kinetcpp/metal
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake

# Optional: OpenMP for CPU parallelization
sudo apt-get install -y libomp-dev

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install
sudo cmake --install build
sudo ldconfig
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install -y gcc-c++ cmake

# Optional: OpenMP
sudo dnf install -y libomp-devel

# Build and install
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### Windows

```powershell
# Using Visual Studio 2019/2022 Developer Command Prompt

# Configure with MSVC
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build Release
cmake --build build --config Release

# Install
cmake --install build --prefix C:\kinet
```

---

## Using the Library

### With CMake (Recommended)

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project)

# Find Kinet GPU
find_package(kinetgpu REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app kinet::kinetgpu_core)
```

Configure your project:

```bash
cmake -B build -Dkinetgpu_DIR=/path/to/kinetgpu/lib/cmake/kinetgpu
cmake --build build
```

### With pkg-config

```bash
# Compile
gcc -o my_app main.c $(pkg-config --cflags --libs kinetgpu)

# Check flags
pkg-config --cflags kinetgpu   # -I/usr/local/include
pkg-config --libs kinetgpu     # -L/usr/local/lib -lkinetgpu -ldl
```

### Direct Linking

```bash
# Linux
gcc -o my_app main.c -I/usr/local/include -L/usr/local/lib -lkinetgpu -ldl

# macOS
clang -o my_app main.c -I/usr/local/include -L/usr/local/lib -lkinetgpu

# Windows (MSVC)
cl /I C:\kinet\include main.c /link C:\kinet\lib\kinetgpu.lib
```

---

## Backend Plugins

The core library includes only the CPU backend. GPU backends are separate plugins.

### Available Backends

| Backend | Repository | Platform | Dependencies |
|---------|------------|----------|--------------|
| Metal | `kinetcpp/metal` | macOS arm64 | Metal.framework, MLX |
| CUDA | `kinetcpp/cuda` | Linux, Windows | CUDA Toolkit 12+ |
| WebGPU | `kinetcpp/webgpu` | All | Dawn or wgpu |

### Installing Backend Plugins

Backend plugins are shared libraries placed in the plugin search path.

**Plugin naming convention**:
- Linux: `libkinetgpu_backend_<name>.so`
- macOS: `libkinetgpu_backend_<name>.dylib`
- Windows: `kinetgpu_backend_<name>.dll`

**Search paths** (in order):
1. `KINET_GPU_BACKEND_PATH` environment variable
2. System library paths (`/usr/lib/kinet-gpu`, `/usr/local/lib/kinet-gpu`)
3. Relative to executable

### Example: Installing Metal Backend

```bash
# Clone Metal backend
git clone https://github.com/kinetfi/metal.git
cd metal

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install to system plugin directory
sudo mkdir -p /usr/local/lib/kinet-gpu
sudo cp build/libkinetgpu_backend_metal.dylib /usr/local/lib/kinet-gpu/
```

### Example: Installing CUDA Backend

```bash
# Requires CUDA Toolkit 12+
git clone https://github.com/kinetfi/cuda.git
cd cuda

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

sudo cp build/libkinetgpu_backend_cuda.so /usr/local/lib/kinet-gpu/
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `KINET_GPU_BACKEND_PATH` | Plugin search path | `/opt/kinet/backends` |
| `KINET_BACKEND` | Force specific backend | `cuda`, `metal`, `cpu` |

### Force Backend Selection

```bash
# Use CPU backend even if GPU available
export KINET_BACKEND=cpu
./my_app

# Use specific plugin path
export KINET_GPU_BACKEND_PATH=/opt/custom/backends
./my_app
```

---

## Verification

### Test Installation

```c
// test_install.c
#include <kinet/gpu.h>
#include <stdio.h>

int main() {
    printf("Kinet GPU version: %d.%d.%d\n",
           KINET_GPU_VERSION_MAJOR,
           KINET_GPU_VERSION_MINOR,
           KINET_GPU_VERSION_PATCH);

    KinetGPU* gpu = kinet_gpu_create();
    if (gpu) {
        printf("Backend: %s\n", kinet_gpu_backend_name(gpu));
        KinetDeviceInfo info;
        if (kinet_gpu_device_info(gpu, &info) == KINET_OK) {
            printf("Device: %s\n", info.name);
        }
        kinet_gpu_destroy(gpu);
        return 0;
    }
    return 1;
}
```

Compile and run:

```bash
gcc -o test_install test_install.c $(pkg-config --cflags --libs kinetgpu)
./test_install
```

Expected output:

```
Kinet GPU version: 0.2.0
Backend: cpu
Device: CPU (SIMD)
```

### Run Test Suite

```bash
cd build
ctest --output-on-failure

# Verbose output
ctest -V

# Run specific test
ctest -R test_gpu_core
```

---

## Troubleshooting

### Library Not Found

```
error: kinetgpu not found
```

**Solution**: Ensure library path is set.

```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### Backend Plugin Not Loading

```
Backend 'metal' not available
```

**Solution**: Check plugin path and permissions.

```bash
# Verify plugin exists
ls -la /usr/local/lib/kinet-gpu/

# Set explicit path
export KINET_GPU_BACKEND_PATH=/usr/local/lib/kinet-gpu
```

### OpenMP Not Found

```
CMake Warning: OpenMP not found
```

**Solution**: Install OpenMP development package.

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp

# Fedora
sudo dnf install libomp-devel
```

### ABI Version Mismatch

```
Backend ABI version mismatch: expected 2, got 1
```

**Solution**: Rebuild backend plugin with matching core library.

```bash
# Check core ABI version
grep KINET_GPU_BACKEND_ABI_VERSION include/kinet/gpu/backend_plugin.h

# Rebuild backend
cd /path/to/backend
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first
```

---

## Development Build

For development with debug symbols and sanitizers:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
    -DKINET_GPU_BUILD_TESTS=ON

cmake --build build
ctest --test-dir build
```

### Code Coverage

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="--coverage"

cmake --build build
ctest --test-dir build

# Generate coverage report
gcov build/CMakeFiles/kinetgpu_core.dir/src/*.cpp.gcno
```
