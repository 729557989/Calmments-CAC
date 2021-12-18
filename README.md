# x-24, youtube comments analytics tool, Name: Calmments

# How to run this
1. go to the path where the folder is located
2. Make sure the django library is accesible under this folder path.
3. to run the website do: python manage.py runserver

-> AI_model folder stores the AI model, tokenizer model, and the file for making predictions for the comments.
-> AI_training_folder is where we trained the AI model, it contains the dataset, a jupyter notebook + pyhton file for training the AI model.
NOTE: the dataset is from Kaggle, link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


-> youtube_comments folder stores the file for scraping youtube comments.
-> have a requirements.txt for all the libraries and version used

# Motivation
Purpose: filter toxic comments on youtube so our users could read all neutral comments

Target Users: open to anyone as this product is intuitive but especially the group of people below:
