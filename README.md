# HimusNN

Simple multi-layer perceptron (MLP) in C++ with manual backprop using matrix-based layers. The example trains a 12-bit parity function or a custom character classification dataset.

## Features

- Matrix-based layers (W*x + b) with batch forward/backprop using `float32` precision
- **Apple Silicon Native**: Hybrid dispatch engine utilizing Apple Accelerate (AMX/vDSP) for CPU and Metal (MPS) for GPU scaling
- **Zero-Copy Memory**: Contiguous, flat memory layout using C++20 `std::span` mapped directly to Metal via `MTLResourceStorageModeShared`
- **Hardware Transposition**: Eliminated memory-copying overhead by utilizing native CBLAS ILP64 (`CblasTrans`) and Metal transposition flags during matrix multiplication
- **CPU Multi-threading**: Native macOS Grand Central Dispatch (`dispatch_apply`) integration for parallel execution of non-linear operations and element-wise $O(N)$ math
- **Batched Evaluation**: Efficient validation and metrics calculation using large batch processing instead of single-vector iteration
- **Adam Optimizer**: Stable and fast convergence using Adam / RMSProp with in-place zero-allocation updates
- **Modern C++20 API**: Enforced `noexcept` move semantics, `[[nodiscard]]` safeguards, `explicit` constructors, and `[[unlikely]]` branch prediction hints for zero-overhead execution
- Modular activations (separate for hidden and output layers) and loss functions
- CSV data loader with train/val split and min-max normalization
- Training progress bar with metrics and ETA (train + validation)
- Training example on binary or categorical data (e.g., EMNIST Alphabet)
- Build with CMake

## Requirements

- CMake >= 3.16
- C++20 compiler
- macOS 13.3+ (for native Accelerate ILP64 API, Metal acceleration, and GCD) or Windows/Linux (with built-in fallback)

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

The program trains a network and prints predictions:

```
Predicted: T' | Confidence: 100.00% OK
Predicted: G' | Confidence: 100.00% OK
Predicted: M' | Confidence: 100.00% OK
Predicted: J' | Confidence: 100.00% OK
```

## Structure

- `src/` - implementations and training loop (including `.mm` Objective-C++ implementation for Metal)
- `include/` - class headers (`Matrix`, `Layer`, `Network`, `Trainer`, `Progress`, `Timing`, `Metric`, `Config`, `DataSet`)
- `CMakeLists.txt` - build configuration with Apple frameworks linking

## Notes

- Training settings (epochs, intervals, learning rate, betas, activations, loss) are defined in `Config`.
- The architecture dynamically chooses CPU (Accelerate) for matrix operations below a complexity threshold, and pushes heavier loads straight to GPU (Metal).
- The dataset is loaded into a flat 1D buffer to eliminate fragmentation and nested vector overhead.
- Normalization stats are computed on the train set and applied to both train and val.
- Legacy neuron-based version is available on the `old-neuron` branch.
