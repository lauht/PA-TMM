# PA-TMM

This is a PyTorch implementation of the paper "Responding to News Sensitively in Stock Attention Networks via Prompt-Adaptive Tri-Modal Model".

## Requirements
* python==3.7.13
* torch==1.8.0

## News
News headlines are collected from [Benzinga](https://github.com/Benzinga/benzinga-python-client).

## How to train the model
1. Run `MPA.ipynb`.
This script would build and pre-train a PA-TMM with the movement prompt adaptation strategy.
2. Run `PA-TMM.ipynb`.
This script would load the pre-trained parameters to the PA-TMM model, and then fine-tune it.
