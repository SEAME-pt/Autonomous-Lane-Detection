# Lane Detection model

The dataset we are using to train the model is from: <https://www.kaggle.com/datasets/manideep1108/culane>. From this link we downloaded the following directories, for the images: driver_161_90frame, for the binary masks: laneseg_label_w16, and for testing: driver_182_30_frame.

## Project Architecture

For real time testing, we are going to use OpenCV to pre-process our images and then send them to the Pytorch model, so that we can get the best and fastest results.

## Pytorch

Why use Pytorch and not TensorFlow?
Pytorch is a more user friendly machine learning framework, and the end result will ultimately be the same between the two. We can also convert the Pytorch model to TensorRT to optimize it for Jetson Nano.

## Neural Networks

Neural networks (especially CNNs) excel at learning spatial hierarchies and feature representations directly from pixel data.
Random Forest might perform well on clean, structured roads but fail on foggy or nighttime roads where lane markings are faint. CNNs, trained on diverse datasets, adapt much better.

## Hyperparameters

We are using the Adam optimizer for training, as it is well-suited for segmentation tasks due to its adaptive learning rate mechanism. The optimizer is initialized with a standard learning rate of 0.001. However, we are also using a learning rate scheduler to dynamically adjust the learning rate if the loss function shows no improvement over a certain number of epochs. We also use a weight decay of 1e-4, this helps preventing overfit.

For the loss function, we combine Focal and Dice Loss. Focal Loss is effective in handling class imbalance by giving more weight to hard-to-classify examples (the incorrect predictions), which is common in segmentation tasks. On the other hand, Dice Loss is great for segmentation because it focuses on the intersection of predicted and ground truth values, rather than the union, making it more sensitive to smaller, harder-to-detect regions.

We are also applying transformations (data augmentation) to the images, which helps the model adapt and generalize to a wider range of input data. Additionally, to combat overfitting, we apply random dropouts in the model's layers during training, which helps prevent the model from depending too much on certain neurons or features, encouraging it to learn more general patterns.

We also use skip connections to ensure our model doesn't lose important information from earlier layers.

nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1):

    This is a 2D convolutional layer that applies a kernel/filter of size 3x3 across the input image. Padding ensures the output has the same spatial dimensions as the input, keeping the feature map size the same after convolution.

nn.ReLU(inplace=True):

    Rectified Linear Unit is a non-linear activation function that helps the model learn complex patterns. It outputs the input if it’s positive, or zero if it’s negative. inplace=True means it modifies the input tensor directly, which helps save memory during training.

nn.MaxPool2d(2):

    Max Pooling reduces the spatial dimensions (height and width) of the input by taking the maximum value from a 2x2 window. It helps reduce the complexity of the model and focuses on the most important features, providing some translation invariance.

Decoder (ConvTranspose2d):

    This layer is the opposite of a convolution, performing upsampling. This helps increase the spatial dimensions of the feature maps and is used in the decoder part of the network to recreate the original image size.

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
python testing.py
```

## Converting

To convert the pytorch model to TensorRt, you need to send to the Jetson Nano the model (.pth), and the convert.py script, you can do this by running:

```bash
chmod +x send_data.sh
send_data.sh
```

After this, run in jetson the converting script, in a virtual environment, be sure to install tensorrt as well:

```bash
python convert.py
```
