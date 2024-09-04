# test-yv8-model

Application script used to test YOLOv8 models against a test video stream.  The YOLOv8 model must be ONNX format.

## Setup
This app requires an alwaysAI account. Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI toolchain on your development machine.

Next, create an empty project to be used with this app. When you clone this repo, you can run `aai app configure` within the repo directory and your new project will appear in the list.

## Usage
Once you have the alwaysAI tools installed and the new project created, run the following CLI commands at the top level of the repo:

To set the project, and select the target device run:

```
aai app configure
```

To build your app and install on the target device:

```
aai app install
```

To start the app:

```
aai app start
```

## Design
This application script is used to test the performance of YOLOv8 models running against a video file input stream.  The ONNX model will execute either on the cpu or can be compiled to TENSORT runtime format depending on which edgeiq engine you choose (Engine.ONNX_RT or Engine.TENSOR_RT).  Future releases of this repo will support batch processing and ONNX runtime using CUDA.

## ONNX Conversion  
