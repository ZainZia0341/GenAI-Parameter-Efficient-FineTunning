POST

🌟 Exciting Journey into Generative AI with Large Language Models! 🌟

I’m thrilled to share that I’ve just completed Week 2 of the "Generative AI with Large Language Models" course by DeepLearning.AI and Amazon Web Services on Coursera! 🚀

In this week's lab, I continued to leverage AWS SageMaker for executing code and delved into advanced topics such as single task and multitask fine-tuning, model evaluation, and parameter efficient fine-tuning (PEFT) techniques. Here's a detailed summary of my Week 2 journey:

For an overview of Week 1 content, check out this post. https://www.linkedin.com/posts/zainzia0341_ai-machinelearning-deeplearning-activity-7198343417637023746-HU5h?utm_source=share&utm_medium=member_desktop

And for the code of Week 1 lab, visit the GitHub repository here. https://github.com/ZainZia0341/GenAI-Zero-shot-and-Few-shot-prompting

📂 GitHub Link Week 2 Lab: .

🔍 Single Task and Multitask Fine-Tuning: Explored the concepts of single task and multitask fine-tuning for natural language processing models, understanding their implications for model performance.

🔄 Model Evaluation and Benchmarks: Learned about different evaluation metrics used to assess the performance of fine-tuned models and compared various benchmark models to gauge effectiveness.

⚙️ Parameter Efficient Fine-Tuning (PEFT): Dived into the concept of PEFT and its importance in optimizing model performance while minimizing computational resources.

🔍 PEFT Technique 1: LoRA - Discovered the Layerwise Learning Rate Adjustment technique for parameter efficient fine-tuning, optimizing learning rates across different layers of the model architecture. LoRA employs a low-rank decomposition method to reduce the number of trainable parameters, which speeds up fine-tuning large models and uses less memory.

🔍 PEFT Technique 2: Soft Prompts - Explored the Soft Prompts technique for parameter efficient fine-tuning, enabling the model to learn from indirect supervision signals. Soft prompts provide a flexible way to incorporate additional information into the training process, leading to improved model performance.

This week's topics expanded upon the fundamental concepts introduced in Week 1 and provided me with advanced techniques for fine-tuning and evaluating models. I'm excited to continue this journey of mastering generative AI!

#AI #MachineLearning #DeepLearning #GenerativeAI #Transformers #NLP #ParameterEfficientFineTuning #Coursera #AWS #DeepLearningAI


Readme file

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
