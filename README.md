# MNIST-Digit-Recognition
This project focuses on the development of a machine learning model designed to recognize and predict handwritten digits using the MNIST dataset

## Overview
This repository contains the code and resources for building a machine learning model to recognize handwritten digits from the MNIST dataset. The goal of this project is to create a robust and accurate model capable of predicting digits from 0 to 9.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is 28x28 pixels in size.

## Project Structure
- `data/`: Contains the dataset (training and test sets).
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model training.
- `src/`: Source code for the project, including data processing and model definition scripts.
- `models/`: Saved models and checkpoints.
- `results/`: Evaluation results and performance metrics.

## Installation
To run this project, you need Python 3.7+ and the following dependencies:

```bash
pip install -r requirements.txt
```
## Usage
Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition
```
## Download the MNIST dataset:
The dataset will be automatically downloaded when you run the notebook or script.

## Run the Jupyter notebook:
```bash
jupyter notebook notebooks/main.ipynb
```
## Train the model:
You can train the model by running the scripts in the src directory or by following the steps in the Jupyter notebook.

## Model
The model used in this project is a convolutional neural network (CNN) implemented using TensorFlow and Keras. It consists of several convolutional layers followed by fully connected layers.

## Run the Application
Start the Flask app:
```bash
python app.py
```
Access the application in your web browser at http://localhost:5000/.

## Results
The final model achieves an accuracy of 98.99% on the MNIST test set.

After running app.py, navigate to http://localhost:5000/ to access a canvas where you can draw digits. Submit your drawing to see how accurately my model predicts the digit.

![image](https://github.com/yugeshsivakumar/MNIST-Digit-Recognition/assets/156910899/d68cf617-228d-4d3d-b93a-a1ca627e2b66)



## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- The creators and maintainers of the MNIST dataset.
- TensorFlow and Keras for providing powerful tools for deep learning.
- The open-source community for continuous support and contributions.
