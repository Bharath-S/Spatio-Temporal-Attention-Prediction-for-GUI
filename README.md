# Spatio Temporal Attention Prediciton



# Introduction

This repository contains code for the re-implementation of the static and dynamic attention prediction models mentioned in the paper. It also contains LSTM based models that extend the base model in static and dynamic predictions

## Usage

The Orchestrator can is used to orchestrate the interactions with the application.
The Orchestrator accepts two parameters: Service name, port name

Generic example:
>cd Spatio_Temporal_Attention_Prediciton
>python3 Orchhestrator.py (Service Name) (Port Name)

#### Supported services:
1. static : References the base Static Attention Prediction model
2. dynamic : References the base Dynamic Attention Prediction model
3. lstm_static : References the LSTM based Static Attention Prediction model
4. lstm_dynamic: References the LSTM based Dynamic Attention Prediction model
#### Supported ports:
1. dataset : creates the processed data required for training
2. train: Train the model
3. test: Test the model ( generate the prediction image )
4. quickeval: Quick evaluation of the model w.r.t the supported evaluation metrics
5. evalLatest: Evaluation performed after generating the dataset, training and prediction

Example, to call create dataset for static attention prediction model: 
> python3 Orchhestrator.py static dataset

To train:
> python3 Orchhestrator.py static train

To test:
> python3 Orchhestrator.py static test

To evaluate:
>python3 Orchhestrator.py static quickeval


## Configuration

The configuration file Config.json allows to configure few of the parameters which is used by the application. It can be extended further on

#### Static parameters:
#### Dynamic parameters
1. FeatureDimension : Dimension of the feature over time. The feature map can be calculated by 
( 2*FeatureDimension + 1)
#### LSTM Static parameters:
#### LSTM Dynamic parameters:


