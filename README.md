# Typhoon Prediction Using Deep Learning

This repository contains the code and resources for our paper on typhoon prediction using deep learning. The project utilizes various deep learning models to predict the presence of typhoons in given datasets.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The goal of this project is to develop a model that can predict the presence of typhoons using meteorological data. We leverage deep learning techniques and specifically use ResNet models for our predictions.

## Dataset

The dataset used in this project consists of meteorological data stored in NetCDF files. The dataset includes variables such as `isobaricInhPa`, `latitude`, `longitude`, `tmptrp`, `landmask`, `rhprs`, `hgttrp`, `time`, `vvelprs`, `pressfc`, `tmpsfc`, `tmpprs`, `vgrdprs`, `ugrdprs`, `hgtprs`, and `absv`.

- Positive samples: `/content/Positive/POSITIVE/*.nc` (contains typhoon)
- Negative samples: `/content/Negative/PastDomain/*.nc` (does not contain typhoon)

## Models

We use various ResNet models for our predictions. The models can be selected by the user at runtime.

## Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
