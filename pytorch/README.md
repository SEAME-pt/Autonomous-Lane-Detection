# Running

To run this code, you need to install all dependencies such as torch, torchvision and pillow (PIL). I advise you to create and activate a virtual python environment to install these libraries and run the code.

# Training
To start training the model, run the following, inside the pytorch directory:

```bash
python train.py
```

# Testing
To test the model, run the following, inside the pytorch directory:

```bash
python test.py
```

# Converting

To convert the pytorch model to tensorrt, you need to send to the jetson nano the model (.pth), and the convert.py script, you can do this my running:

```bash
chmod +x send_data.sh
send_data.sh
```

After this, run in jetson the converting script, in a virtual environment, be sure to install tensorrt as well:

```bash
python convert.py
```