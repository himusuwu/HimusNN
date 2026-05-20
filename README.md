# HimusNN

Simple multi-layer perceptron (MLP) in C++ with manual backprop. The example trains a 3-bit parity function.

## Features

- Implementation of neurons, layers, and network (forward + backprop)
- Training example on binary data
- Build with CMake

## Requirements

- CMake >= 3.16
- C++20 compiler

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

The program trains a `3 -> 6 -> 1` network for 50_000 epochs and prints predictions:

```
000 -> 0 (raw=...)
001 -> 1 (raw=...)
010 -> 1 (raw=...)
011 -> 0 (raw=...)
100 -> 1 (raw=...)
101 -> 0 (raw=...)
110 -> 0 (raw=...)
111 -> 1 (raw=...)
```

## Structure

- `src/` - implementations and the example in `main.cpp`
- `include/` - class headers: `Neuron`, `Layer`, `Network`
- `CMakeLists.txt` - build configuration

## Notes

- The learning rate is set to `0.5` in `Network`.
- Training data and epoch count are in `main.cpp`.
