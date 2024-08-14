<a id="readme-top"></a>




# Assignment-3-LLM

A repository contains my work for Assignment 3 of Research Methods, part of our coursework at the [University of Hertfordshire, UK](https://www.herts.ac.uk/).

## Project Overview



<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Introduction

In the vast digital landscape of social media, discussion forums, and comment sections, toxicity poisons conversations, drives away users, and harms online communities. As we navigate this interconnected world, identifying and mitigating toxic content becomes increasingly critical.
This assignment delves into precisely this challenge and leverages Large Language Models (LLM) to predict whether a given comment is toxic or not.
The task is a binary classification: given a comment, determine whether it exhibits toxic behaviour.

The motivation for developing an automated toxicity classifier is to help create safer virtual spaces and improve user experience. Toxic comments, ranging from hate speech to personal attacks, silence voices, intimidate, and drive away genuine contributors. Swiftly flagging toxicity will empower individuals and contribute to creating a more respectful digital environment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Overview

The Wiki Toxic dataset serves as a modified and cleaned version of the original dataset used in the Kaggle Toxic Comment Classification Challenge (conducted during 2017/18). It comprises comments collected from Wikipedia forums and is labelled into two distinct categories: toxic and non-toxic. The dataset is obtained from Hugging Face, a reputable platform for sharing and accessing natural language processing (NLP) datasets and pre-trained models.

The dataset is well-suited for text classification tasks, particularly for training models to identify toxicity in sentences and classify them accordingly.
All data instances in the dataset are in English language. Each data instance comprises the following components:

- id: A unique identifier string associated with each comment.
- comment_text: The actual textual content of the comment as a string.
- label: An integer label, 0 for non-toxic and 1 for toxic comments.

The dataset is divided into several subsets: a train set containing 127,656 instances, a validation set containing 31,915, a test set containing 63,978 and a balanced_train set, a subset of the training data with 25,868 instances, balanced in terms of toxic and non-toxic comments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Exploratory Data Analysis



<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Pre-Processing

Prior to commencing model training, we thoroughly preprocess the Wiki Toxic dataset.
The balanced_train dataset is separated into training and validation subsets.
Due to our restricted computational resources, we choose to use this smaller subset and its inherent balance simplifies our task.
A validation size of 30% is used and the random seed is set up to guarantee reproducibility.


Leveraging the Hugging Face Transformers library, the tokenizer associated with our chosen model is utilized to convert raw comment text into tokenized input suitable for the model.
The tokenizer is verified to be a fast tokenizer and its maximum token length is determined.
A custom function, tokenize is defined, which also truncates comments to a maximum length of 128 tokens to run our model on GPUs found on Google Colab. .map method is used to tokenize the whole dataset with the tokenize fucntion.

To ensure a consistent input format for training and validation, a data collator is defined that handles padding.

Tokenized training and validation data is converted into TensorFlow datasets using .to_tf_dataset with the defined data collator and batch size of 16.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model



<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results and Discussion

The model achieved an accuracy of 92% during both training and validation, indicating that it learned the classification task well on balanced datasets. On the imbalanced test set, which contains fewer toxic comments, the model achieved a high Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) score of 0.9552. This indicates that the model has a high overall ability to distinguish between toxic and non-toxic comments.

The classification report shows that the model performs exceptionally well in the non-toxic class, with high precision. However, the recall for the non-toxic class is significantly lower, indicating that the model occasionally misclassifies non-toxic comments as toxic.

In contrast, for the toxic class, the model has a low precision of 0.39, indicating that a significant number of non-toxic comments are incorrectly classified as toxic. This points to a higher rate of false positives. Despite this, the recall for the toxic class is notably high, at 0.94, indicating that the model is extremely effective at identifying the most toxic comments.

These results indicate that while the model performs well on balanced data, its generalization to imbalanced data is limited. The low precision for the toxic class on the imbalanced test set suggests a tendency for false positives, possibly due to the model being overly confident in its toxic predictions. Despite this, the high recall demonstrates the model’s effectiveness in capturing toxic content.

In this context, the trade-off between precision and recall may be acceptable, as misclassifying non-toxic comments as toxic might be more tolerable than failing to identify actual toxic comments. However, this trade-off should be carefully considered, particularly in applications where the cost of false positives is high.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Conclusion



<p align="right">(<a href="#readme-top">back to top</a>)</p>

I hope you find this project insightful and useful!
