# Typhoon Prediction Using Deep Learning

This repository contains the code and resources for our paper on typhoon prediction using deep learning. The project utilizes various deep learning models to predict the presence of typhoons in given datasets.

## Table of Contents

- [Introduction](#introduction)
- [Models](#models)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Introduction

The goal of this project is to develop a model that can predict the presence of typhoons using meteorological data. We leverage deep learning techniques and specifically use ResNet models for our predictions.

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
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the test set:
The test set will be automatically downloaded when running the main script if it does not exist.

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
resnet_t16.h5.

For example, to use resnet_t2.h5, you would run:
```bash
python main.py --model resnet_t2.h5
```

This command will:
- Load the specified model.
- Check if the test set file already exists. If not, it will create the necessary directories and download the file.
- Load the test set into test_set.
- Create labels y with a shape matching the number of samples in the test set.
- Calculate precision, recall, and F1 score for both labels.
- Write these metrics to ./result/result.txt.
- Print a confirmation message that the metrics have been calculated and written to the file.

## Results
The results of the model's performance, including precision, recall, and F1 score for both labels, will be written to ./result/result.txt.

