# x-24, youtube comments analytics tool, Name: Calmments

# How to run this -----------------------------------------------------------------------------------------------------
1. go to the path where the folder is located
2. Make sure the django library is accesible under this folder path.
3. to run the website do: python manage.py runserver

-> AI_model folder stores the AI model, tokenizer model, and the file for making predictions for the comments.
-> AI_training_folder is where we trained the AI model, it contains the dataset, a jupyter notebook + pyhton file for training the AI model.
NOTE: the dataset is from Kaggle, link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


-> youtube_comments folder stores the file for scraping youtube comments.
-> have a requirements.txt for all the libraries and version used

# Motivation-----------------------------------------------------------------------------------------------------------
Purpose: filter toxic comments on youtube so our users could read all neutral comments

Target Users: open to anyone as this product is intuitive but especially the group of people below:

1. students
    Why? 
        Teachers sometimes have youtube videos as extra resources for students to learn.
        In turn, students also read the comments under these videos for views from different perspectives
        However, some comments are really racist and toxic. (especially history videos for some reason...)
        So, we aim to filter these toxic comments so our user only sees the neutral ones where they feel more comfortable reading.

2. youtube vloggers
    Why?
        Vloggers often read comments to see audiences' reaction to their videos
        However, some are purely toxic and insulting where there's absolutely no reason for anyone to read it.
        So, we help them filter these comments with the aid of our product.
        Additionally, if the vloggers really do want to sort through all the negative comments(despite the toxicity) and see what people are all complaining about. Then, they could also view all the negative comments and analyze them.

3. General comment readers
    Why?
        We usually watch comments for entertainment, but some are so toxic that makes us feel unhappy and uncomfortable. So, our product filters the negative comments so people could enjoy reading comments.