## Project Architecture

A simple explanation of our structure.

<!-- ![Our project structure](ADR/structure.png) -->
\image html ADR/structure.png "Project Structure" width=50%

If you want to see more on our LaneNet model, please check out [Model README](/pytorch/README.md).

## Results

These images were captured during training with TUSimple and CULane datasets, after a certain amount of epochs.
<!-- 
![](results/tusimple.png) -->
\image html results/tusimple.png "Training" width=50%

This image is in epoch 37, nightime.

<!-- ![](results/epoch37.png) -->
\image html results/epoch37.png "Training" width=50%

This image was taken whilst testing the model in Jetson. Here you see different masks, with different thresholds of values, after applying the activation function **sigmoid**. After removing the fisheye effect of Jetson's camera, the model produced much better results.

<!-- ![](results/jetson_model.jpeg) -->
\image html results/jetson_model.jpeg "Jetson" width=50%

Next there are a few videos we recorded while testing the model in CARLA. On one side, you see a CARLA's window, with the vehicle from a top spectator view. On the other side, you see an **OpenCV** window from the car's perspective, with our pytorch model's binary mask overlaying the road.
In intersections, because there is NO lane, we defined that the car should go straight ahead, in the CARLA environment. You can see this behaviour in *carla_setup.py*

Click to see [Town5 Demo](/pytorch/results/town5.mp4), and [Town4 Demo](/pytorch/results/town4.mp4).


## Chosen datasets
The datasets we are using to train the model are from: [This is an external link to the Datasets used](https://onedrive.live.com/?id=4EF9629CA3CB4B5E%213022&cid=4EF9629CA3CB4B5E&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbDVMeTZPY1l2bE9sMDQxNHNSb3BGVkgyOTVXP2U9Q2pjbDYy). We used dataset8 and dataset10. However, we were using TUSimple and CULane beforehand, but couldn't get good results on CARLA, so we decided to switch datasets. For testing, we used dataset11 and **CARLA**. We have a testing script for CARLA in *carla_setup.py*.

## Pytorch Model

We use **OpenCV** to pre and post-process our images, sending them to our Pytorch model. This way we can get the correct results. We also have a *retrain.py* file to train the model even more after its already been saved. The learning rate this way, is a little bit lower than the previous training.
Why use Pytorch and not TensorFlow?
Pytorch is a more user friendly machine learning framework, and the end result will ultimately be the same between the two. We can also convert the Pytorch model to TensorRT to optimize it for Jetson Nano.
Now a little bit on **Neural Networks**, Neural networks (especially **CNNs**) excel at learning spatial hierarchies and feature representations directly from pixel data.
Random Forest might perform well on clean, structured roads but fail on foggy or nighttime roads where lane markings are faint. CNNs, trained on diverse datasets, adapt much better.

## Hyperparameters

Change the **Batch Size** according to your Gpu capabilities, in *dataset.py*.

We are using the **Adam Optimizer** for training, as it is well-suited for segmentation tasks due to its adaptive learning rate mechanism. The optimizer is initialized with a standard learning rate of 0.001. However, we are also using a **learning rate scheduler** to dynamically adjust the learning rate if the loss function shows no improvement over a certain number of epochs. We use a **weight decay** of 1e-4, this helps preventing overfit.

For the loss function, we combine **Focal and Dice Loss**. Focal Loss is effective in handling class imbalance by giving more weight to hard-to-classify examples (the incorrect predictions), which is common in segmentation tasks. On the other hand, Dice Loss is great for segmentation because it focuses on the intersection of predicted and ground truth values, rather than the union, making it more sensitive to smaller, harder-to-detect regions.

Additionally, to combat overfitting, we apply random **Dropouts** in the model's layers during training, which helps prevent the model from depending too much on certain neurons or features, encouraging it to learn more general patterns. We also apply many **Transformations** to the images, such as Horizontal Flip, Motion and Gaussian blur, and color alterations.

We use **Skip Connections** to ensure our model doesn't lose important information from earlier layers.