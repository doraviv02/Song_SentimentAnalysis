
# Fine-Tuning a BERT Model For Regressive Song Sentiment Analysis

Fine-tuning of BERT model for the purposes of Sentiment Analysis

<img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Concept.png" width="800">

## Table of Contents

- Introduction
- Method
- Experiments & Results
- Conclusions & Future Work
- How to run
- References

## Introduction

Our project focuses on sentiment analysis in the music domain, leveraging the power of Natural Language Processing (NLP). By analyzing song lyrics, we gain valuable insights for tasks like music recommendation and playlist generation. Our approach involves fine-tuning a BERT model, known for its contextual understanding, making it ideal for sentiment analysis tasks.

To tackle the computational challenges, we employ the Low Rank Adaptation (LoRA) technique, which streamlines the training process for large models like BERT by reducing the parameter space. Additionally, we utilize Weights & Biases (wandb) for hyperparameter optimization, enhancing the model's performance through systematic parameter tuning.


## Method

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is an NLP model developed by Google. The main innovation of BERT is its capability of bidirectional understanding of textual contexts. BERT can capture the nuances and complexities of language by considering both preceding and succeeding words in a sentence, enabling it to generate highly accurate  representations of words. 

### LoRA
<p float="center">
  <img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/architecture.png" width="500">
</p>

LoRA (Low Rank Adaptation) is a technique used to streamline the training process for large models like BERT in the context of sentiment analysis. It reduces the parameter space of the model, making it more computationally efficient without sacrificing performance. By leveraging low-rank matrix factorization, LoRA effectively compresses the model while preserving its contextual understanding capabilities. This allows for faster training and inference times, making it ideal for large-scale sentiment analysis tasks.

### Loss Function
Since we chose to fine-tune over a regression task, we used the MSE loss function to train our model.

### Weights & Biases
Weights and biases is a powerful modern platform for monitoring and analyzing models. It offers extensive visualizations that allow for effective experiment tracking and analyzing. 
In our project, we utilized W\&B's capabilities to run a Bayesian hyperparameter sweep, giving us a better estimation of which hyperparameters to use.  

The idea behind a Bayesian hyperparameter search is to build a probabilistic model for our objective function of the form:

### Hugging Face 🤗

Hugging Face is a framework for building applications using machine learning systems. The framework is especially optimized for NLP tasks and specifically working with Transformers. We used hugging face throughout the entire project - from importing the pretrained BERT model, integrating the LoRA method, and connecting to wandb in order to perform the hyperparameter tuning. The (relative) simplicity of using hugging face allowed to abstract the fine details on focus on the high-level concepts we were trying to implement in our project.


## Experiments and Results

### Datasets

The dataset we used was specifically tailored for song sentiment analysis. It is
composed of 150k song lyrics, author name, and their valence parameter, which
is a measurement of the songs positiveness that is computed using Spotify’s
algorithm. We only needed the lyrics and their valence.
For the purposes of our project we took a subset of the data: For the hyperparameter tuning we took 10% of the data and split it to training/validation/test
with a ratio of 60/20/20. A small subset allowed us to perform many tests in a
reasonable amount of time.
For the final training of the model we chose to take 40% of the data, around
60k songs, and divide them up to train/test with a ratio of 80/20.

### Hyperparameter Tuning:

<img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Hyperparameters.png" width="1000">

Running 12 trials of tuning while adjust the learning rate, weight decay, scheduler and LoRA rank.

We chose the best trial, which did not display early overfitting and trained using his parameters for 12 epochs:

<img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Training_Model.png" width="1000">


Comparing our architecture with a frozen BERT model with trained regression head:

<p float="left">
  <img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Comparing_Train.png" width="410">
  <img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Comparing_Test.png" width="410">
</p>


## Conclusion and Future Work
Our results provide some interesting outlooks:
 - There is a very weak correlation (compared to other parameters) between the LoRA rank and the model's performance. This indicates that although the LoRA module is effective at improving the training process, low rank is sufficient.
 - We observe overfitting at around the 10th epoch, even after hyperparameter optimization. Further testing could provide insight.
 - Our model is an improvement to the pretrained BERT model, but there is still much to be done.



There is a lot to be done in regard to improvements to our model in order to better perform the sentiment analysis task, as well as many interesting experiments. Some suggestions are:
 - Analysis of the over-fitting runs and finding their common characteristics.
 - Using the analysis provided by W\&B, re-run the hyperparameter search and replace weakly-correlated hyper hyperparameters with some that might have a stronger correlation (for example LoRA's alpha and dropout parameter).
 - Our final model's RMSE on the test set was. An interesting experiment is to quantize our regression model result (i.e. take the regression result and assign a label based on where it falls between 0 and 1) based on this RMSE and compare them to a model trained specifically for classification. We would expect to get good results if the classification model's number of labels N satisfies: $$N = \frac{1}{RMSE}$$

## How to Run
The notebook is divided into sections. Depending on the use case, run each section independently. You can run hyperparameter Sweep (optimization), training, and inference.


- "Setting up notebook", "Imports" and "Setting up WandB" are mandatory for every use of the notebook.

- Every Section, apart from the mandatory ones, contains a configuration subsection where you can configure the parameters as you wish.

- "Loading + Tokenizing Dataset" is mandatory for running the hyperparmeter sweeps

- This was created, and meant to run on "Google Colab", changing environments may require adjustments.

- The code is set up in a way that automatically downloads the required dependencies, but to this manually, run the following command:

```
pip install transformers datasets evaluate accelerate peft wandb
```

You can use our trained model weights for BERT with no LoRA (only the regressive classifier was trained), and BERT with LoRA:
(https://drive.google.com/drive/folders/1hXvFDb-d3EOB49B7-f4xJEoWPxkVHAL3?usp=sharing)



## 🔗 References
[1] [150K Lyrics Labeled with Spotify Valence](https://www.kaggle.com/datasets/edenbd/150k-lyrics-labeled-with-spotify-valence) 

[2] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) 

[3] [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf) 

[4] [Hugging Face - Text Classification](https://huggingface.co/docs/transformers/en/tasks/sequence\_classification)

## Authors

- [@Dor Aviv](https://www.github.com/doraviv02)
- [@Zohar Milman](https://www.github.com/ZoharMilman)

