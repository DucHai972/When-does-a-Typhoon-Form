# Typhoon Prediction Using Deep Learning

This repository contains the code and resources for our paper on typhoon prediction using deep learning. The project utilizes various deep learning models to predict the presence of typhoons in given datasets.

## Table of Contents

- [Introduction](#introduction)
- [Test set](#test-set)
- [Models](#models)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Introduction

The goal of this project is to develop a model that can predict the presence of typhoons using meteorological data. We leverage deep learning techniques and specifically use ResNet models for our predictions.

## Test set

We choose one typhoon out of 80 typhoons in the test set to portray an example. The details of the chosen typhoon are as follows:

- **Event Date**: September 12, 2019
- **Location**: Approximately 9°N latitude and 162°E longitude
- **Time Steps**: The dataset includes 41 time steps, starting from the time the typhoon occurs (t0) and going back to 40 time steps before the event (t-40). Each time step represents a 6-hour interval.

The data is stored in NetCDF (.nc) files, a standard format for storing multi-dimensional scientific data. For our project, we retrieve the data from these NetCDF files and convert it into numpy arrays for easier manipulation and preprocessing before feeding it into our model for prediction.

## Models

We use various ResNet models for our predictions. The models can be selected by the user at runtime.

## Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Setups
1. Clone the repository:
```bash
git clone https://github.com/DucHai972/When-does-a-Typhoon-Form.git
cd When-does-a-Typhoon-Form
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the test set and model:
The test set and model will be automatically downloaded when running the main script if it does not exist.

## Usage
To run the main script and select a model, use the following command:
```bash
python main.py --model <model_name>
```
Replace <model_name> with one of the following options:
resnet_t2.h5,
resnet_t4.h5,
resnet_t5.h5,
resnet_t8.h5,
resnet_t12.h5,
resnet_t16.h5,
cnn_t5.h5.

For example, to use resnet_t2.h5, you would run:
```bash
python main.py --model resnet_t2.h5
```

This command will:
- Load the specified model.
- Verify the existence of the test set file. If it doesn't exist, the command will create the necessary directories and download the file.
- Load the test set into test_set and preprocess it (feature enrichment and normalization).
- Create labels y with a shape matching the number of samples in the test set.
- Calculate precision, recall, and F1 score for the labels.
- Write the evaluation metrics to the ./result/ directory.
- Print a confirmation message indicating that the metrics have been calculated and saved to the file.

## Results
The results of the model's performance, including precision, recall, and F1 score for both labels, will be saved to ./result/.

