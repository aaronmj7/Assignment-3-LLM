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



I hope you find this project insightful and useful!
