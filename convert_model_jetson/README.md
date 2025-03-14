

``` bash
trtexec --onnx=yolov5n.onnx         --saveEngine=yolov5n.trt         --fp16         --workspace=2048         --useSpinWait         --streams=2         --tacticSources=-CUBLAS
```

trtexec
This is NVIDIA’s tool used to convert an ONNX model into TensorRT and measure performance.


2️⃣ --onnx=yolov5n.onnx
Specifies the input ONNX model (yolov5n.onnx).
YOLOv5n (nano) is the lightest version of YOLOv5, optimized for embedded devices.


3️⃣ --saveEngine=yolov5n.trt
Saves the converted model in TensorRT format (.trt).
This model can be loaded directly for optimized inference.


4️⃣ --fp16
Uses FP16 (16-bit precision) instead of FP32 (32-bit), which:
Reduces memory consumption 🚀
Speeds up inference 🔥
Might slightly reduce precision 📉


5️⃣ --workspace=2048
Allocates 2048 MB of memory for optimization.
A larger workspace may improve performance, but uses more RAM.


6️⃣ --useSpinWait
Enables an active waiting method to synchronize threads.
Can improve latency in embedded systems, such as the Jetson Nano.


7️⃣ --streams=2
Specifies 2 parallel inference streams.
Can improve performance if enough computational capacity is available.


8️⃣ --tacticSources=-CUBLAS
Disables CUBLAS (CUDA matrix operations library).
TensorRT may use other tactics to find the fastest solution.