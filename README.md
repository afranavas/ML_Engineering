# CIFAR-10 Image Classification with TensorFlow

This repository contains code for training a convolutional neural network (CNN) model to classify images from the CIFAR-10 dataset. Additionally, it provides a Flask web application for deploying the trained model as a simple API.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is automatically downloaded when running the training script.

## Model Architecture
The model architecture used for this project is a convolutional neural network (CNN) with the following layers:
- Convolutional layer (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling layer (2x2 pool size)
- Convolutional layer (64 filters, 3x3 kernel, ReLU activation)
- MaxPooling layer (2x2 pool size)
- Convolutional layer (64 filters, 3x3 kernel, ReLU activation)
- Flatten layer
- Dense layer (64 units, ReLU activation)
- Dropout layer (50% dropout rate)
- Output Dense layer (10 units, softmax activation)

## Training
To train the model, run the following command:
```bash
python train.py
```
This will train the model on the CIFAR-10 dataset and save the trained model weights.

## Evaluation
After training, you can evaluate the model's performance using the test set by running:
```bash
python evaluate.py
```

## Deployment
To deploy the trained model as an API, run the following command:
```bash
python app.py
```
This will start a Flask web server locally. You can then send POST requests to `http://localhost:5000/predict` with image data to get predictions from the model.

## Demo
You can access a live demo of the deployed model at [Demo Link](#) (replace with your actual demo link).

## Requirements
- Python 3.x
- TensorFlow
- Flask

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
