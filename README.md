# HimusNN

Simple multi-layer perceptron (MLP) in C++ with manual backprop using matrix-based layers. The example trains a 12-bit parity function.

## Features

- Matrix-based layers (W*x + b) with batch forward/backprop using `float32` precision
- **Apple Silicon Native**: Hybrid dispatch engine utilizing Apple Accelerate (AMX/vDSP) for CPU and Metal (MPS) for GPU scaling
- **Adam Optimizer**: Stable and fast convergence using Adam / RMSProp
- Modular activations (separate for hidden and output layers) and loss functions
- CSV data loader with train/val split and min-max normalization
- Training progress bar with metrics and ETA (train + validation)
- Training example on binary data
- Build with CMake

## Requirements

- CMake >= 3.16
- C++20 compiler
- macOS (for native Accelerate and Metal acceleration) or Windows/Linux (with built-in fallback)

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/HimusNN
```

You can also run the CMake target:

```bash
cmake --build build --target run
```

## Example Output

The program trains a `12 -> 32 -> 16 -> 1` network and prints predictions:

```
000000000000 -> 0 (raw=...)
000000000001 -> 1 (raw=...)
000000000010 -> 1 (raw=...)
000000000011 -> 0 (raw=...)
000000000100 -> 1 (raw=...)
000000000101 -> 0 (raw=...)
000000000110 -> 0 (raw=...)
000000000111 -> 1 (raw=...)
```

## Structure

- `src/` - implementations and training loop (including `.mm` Objective-C++ implementation for Metal)
- `include/` - class headers (`Matrix`, `Layer`, `Network`, `Trainer`, `Progress`, `Timing`, `Metric`, `Config`)
- `CMakeLists.txt` - build configuration with Apple frameworks linking

## Notes

- Training settings (epochs, intervals, learning rate, betas, activations, loss) are defined in `Config`.
- The architecture dynamically chooses CPU (Accelerate) for matrix operations below 50M operations, and pushes heavier loads straight to GPU (Metal).
- The example uses a 12-bit parity dataset in `data/parity.csv` and splits it into train/val.
- Normalization stats are computed on the train set and applied to both train and val.
- Legacy neuron-based version is available on the `old-neuron` branch.
