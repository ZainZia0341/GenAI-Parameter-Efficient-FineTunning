# Generative AI for Dialogue Summarization with PEFT

## Introduction
This repository contains the code for the "Generative AI for Dialogue Summarization with PEFT" lab. In this lab, you will learn how to fine-tune a large language model (LLM) for dialogue summarization using both full fine-tuning and Parameter Efficient Fine-Tuning (PEFT) techniques. The lab utilizes the FLAN-T5 model from Hugging Face and evaluates the summarization quality using ROUGE metrics.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
  - [Kernel Setup and Dependencies](#kernel-setup-and-dependencies)
  - [Dataset and Pre-trained Model](#dataset-and-pre-trained-model)
- [Full Fine-Tuning](#full-fine-tuning)
  - [Preprocessing the Dataset](#preprocessing-the-dataset)
  - [Fine-Tuning the Model](#fine-tuning-the-model)
  - [Qualitative Evaluation](#qualitative-evaluation)
  - [Quantitative Evaluation](#quantitative-evaluation)
- [Parameter Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
  - [Setting up PEFT Model](#setting-up-peft-model)
  - [Training PEFT Adapter](#training-peft-adapter)
  - [Qualitative Evaluation (PEFT)](#qualitative-evaluation-peft)
  - [Quantitative Evaluation (PEFT)](#quantitative-evaluation-peft)
- [Conclusion](#conclusion)
- [References](#references)

## Overview
In this lab, you will explore two approaches to fine-tuning a large language model for dialogue summarization: full fine-tuning and PEFT. Full fine-tuning involves training the entire model, while PEFT focuses on training a smaller adapter layer, resulting in reduced computational resources and memory usage.

## Setup

### Kernel Setup and Dependencies
- Ensure the correct kernel is selected.
- Install required dependencies using pip.

### Dataset and Pre-trained Model
- Load the dialogue-summarization dataset from Hugging Face.
- Load the FLAN-T5 pre-trained model for dialogue summarization.

## Full Fine-Tuning

### Preprocessing the Dataset
- Tokenize the dialog-summary pairs and preprocess the dataset.

### Fine-Tuning the Model
- Train the model using the preprocessed dataset.
- Evaluate the model qualitatively and quantitatively.

## Parameter Efficient Fine-Tuning (PEFT)

### Setting up PEFT Model
- Configure the PEFT model with LoRA parameters.
- Add adapter layers to the original LLM.

### Training PEFT Adapter
- Train the adapter layer using PEFT technique.
- Save the PEFT model for future use.

### Qualitative Evaluation (PEFT)
- Evaluate the PEFT model qualitatively with human evaluation.

### Quantitative Evaluation (PEFT)
- Compute ROUGE scores for the PEFT model.
- Compare the PEFT model's performance with the original model.

## Conclusion
This lab demonstrates the effectiveness of PEFT in fine-tuning large language models for dialogue summarization tasks. By leveraging PEFT, practitioners can achieve comparable summarization quality while reducing computational and memory requirements.

## References
- Hugging Face Transformers Documentation: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- Deep Learning AI Courses on Coursera: [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning)
