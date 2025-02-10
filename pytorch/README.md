# Lane Detection model

The dataset we are using to train the model is from: https://www.kaggle.com/datasets/manideep1108/culane. From this link we downloaded the following directories, for the images: driver_161_90frame, for the binary masks: laneseg_label_w16, and for testing: driver_182_30_frame.

## Project architecture

For real time testing, we are going to use OpenCV to pre-process our images and then send them to the Pytorch model, so that we can get the best and fastest results.

## Pytorch

Why use Pytorch and not TensorFlow?
Pytorch is a more user friendly machine learning framework, and the end result will ultimately be the same between the two. We can also convert the Pytorch model to TensorRT to optimize it for Jetson Nano.

## Neural Networks

Neural networks (especially CNNs) excel at learning spatial hierarchies and feature representations directly from pixel data.
Random Forest might perform well on clean, structured roads but fail on foggy or nighttime roads where lane markings are faint. CNNs, trained on diverse datasets, adapt much better.

## Running

To run this code, you need to install all dependencies such as torch, torchvision and pillow (PIL). I advise you to create and activate a virtual python environment to install these libraries and run the code.

## Training

To start training the model, run the following, inside the pytorch directory. We advise you to get access to a gpu:

```bash
python train.py
```

## Testing

To test the model, run the following, inside the pytorch directory:

```bash
python test.py
```

## Converting

To convert the pytorch model to TensorRt, you need to send to the Jetson Nano the model (.pth), and the convert.py script, you can do this my running:

```bash
chmod +x send_data.sh
send_data.sh
```

After this, run in jetson the converting script, in a virtual environment, be sure to install tensorrt as well:

```bash
python convert.py
```
