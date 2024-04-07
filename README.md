
# Fine-Tuning a BERT Model For Regressive Song Sentiment Analysis

Fine-tuning of BERT model for the purposes of Sentiment Analysis

<img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/Concept.png" width="800">

## Table of Contents

- Introduction
- Method
- Experiments & Results
- Conclusions & Future Work
- How to run
- Ethics
## Introduction


Our project focuses on sentiment analysis in the music domain, leveraging the power of Natural Language Processing (NLP). By analyzing song lyrics, we gain valuable insights for tasks like music recommendation and playlist generation. Our approach involves fine-tuning a BERT model, known for its contextual understanding, making it ideal for sentiment analysis tasks.

To tackle the computational challenges, we employ the Low Rank Adaptation (LoRA) technique, which streamlines the training process for large models like BERT by reducing the parameter space. Additionally, we utilize Weights & Biases (wandb) for hyperparameter optimization, enhancing the model's performance through systematic parameter tuning.


## Method

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is an NLP model developed by Google. The main innovation of BERT is its capability of bidirectional understanding of textual contexts. BERT can capture the nuances and complexities of language by considering both preceding and succeeding words in a sentence, enabling it to generate highly accurate  representations of words. 

### LoRA

<img src="https://github.com/doraviv02/Song_SentimentAnalysis/blob/main/images/architecture.png" width="400">

LoRA (Low Rank Adaptation) is a technique used to streamline the training process for large models like BERT in the context of sentiment analysis. It reduces the parameter space of the model, making it more computationally efficient without sacrificing performance. By leveraging low-rank matrix factorization, LoRA effectively compresses the model while preserving its contextual understanding capabilities. This allows for faster training and inference times, making it ideal for large-scale sentiment analysis tasks.

### Loss Function
Since we chose to fine-tune over a regression task, we used the MSE loss function to train our model.

### Weights & Biases
Weights and biases is a powerful modern platform for monitoring and analyzing models. It offers extensive visualizations that allow for effective experiment tracking and analyzing. 
In our project, we utilized W\&B's capabilities to run a Bayesian hyperparameter sweep, giving us a better estimation of which hyperparameters to use.  

The idea behind a Bayesian hyperparameter search is to build a probabilistic model for our objective function of the form:

### Hugging Face 🤗

Hugging Face is a framework for building applications using machine learning systems. The framework is especially optimized for NLP tasks and specifically working with Transformers. We used hugging face throughout the entire project - from importing the pretrained BERT model, integrating the LoRA method, and connecting to wandb in order to perform the hyperparameter tuning. The (relative) simplicity of using hugging face allowed to abstract the fine details on focus on the high-level concepts we were trying to implement in our project.


## Ethics

### Stake Holders
Musicians, Streaming Services, and listeners.
### Implications
Musicians may dislike their music being used for training and regression without their explicit consent. They may also dislike that their music, an artistic expression, is being quantified into such a scale.

Streaming services may enjoy the use of such algorithms in order to automatically assign a valence factor to new songs coming into the service, without Relying solely on user input. On the other hand, they might also dislike the use of their data to train the model without explicit consent.

Listeners may appreciate the ability of streaming services to use the model in order to automatically recommend songs that align with their mood (such as an auto generated playlist of happy music). On the other hand this may disturb their listening experience and feel like it is a breach to their private listening habits.

End-users of musical applications such as Spotify, Apple Music, etc. may enjoy a listening experience better tuned to their expectations of happy/sad music, but it may come to the detriment of the musicians who may not wish to have their songs categorized to such scale.

### Ethical Considerations

Respecting intellectual property rights of artists in the field. Ensuring transparency between streaming services and both artists -  for transparent use of their data, and users - for maintaining their privacy in data collection and registration of their listening habits.
## Experiments and Results

### Datasets

The dataset we used was specifically tailored for song sentiment analysis. It is
composed of 150k song lyrics, author name, and their valence parameter, which
is a measurement of the songs positiveness that is computed using Spotify’s
algorithm. We only needed the lyrics and their valence.
For the purposes of our project we took a subset of the data: For the hyperpa-
rameter tuning we took 10% of the data and split it to training/validation/test
with a ratio of 60/20/20. A small subset allowed us to perform many tests in a
reasonable amount of time.
For the final training of the model we chose to take 40% of the data, around
60k songs, and divide them up to train/test with a ratio of 80/20.





## Conclusion and Future Work


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2(https://twitter.com/)


## Authors

- [@Dor Aviv](https://www.github.com/doraviv02)
- [@Zohar Milman](https://www.github.com/ZoharMilman)

