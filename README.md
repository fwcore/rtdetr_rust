# rtdetr_rust
2024 SOTA object detector inference using Rust + Ort

![](asserts/demo.png)

Currently, I only test ONNX-runtime with CPU backend, without optimization. The latency (including image resize and neural network inference) is about 565ms.

## Get start
(WIP)
* Setup Rust compilier and compile the code
  ```
  cargo build --release
  ```
* Prepare RT-Detr ONNX and an image to detect objects
* Download `libonnxruntime` (please choose the right library according to your OS and platform. Here I use Ubuntu on x86_64 without GPU as an exaxmple):
  ```
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz
  tar xzvf onnxruntime-linux-x64-1.19.0.tgz
  ln -s $(pwd)/onnxruntime-linux-x64-1.19.0/lib/libonnxruntime.so.1.19.0 libonnxruntime.so
  ```
* Run
  ```
  ORT_DYLIB_PATH=../../libonnxruntime.so ./target/release/rtdetr_rust path/to/your/image
  ```
