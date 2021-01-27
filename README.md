# Recommendation systems

The present repository contains the python3 implementation of 2 different recommendation systems where sparse records of movie ratings are leveraged to produce predictions for the level of satisfaction a user is expected to report in relation to a movie he/she has not rated before. The algorithms are built upon the concept of collaborative filtering such that the prediction of a missing rating for a user is produced through a model trained on a large dataset of historical rating records.
The setting we adopted for the implementation was inspired by the challenge launched by Netflix in 2006 which prompted data scientists worldwide to work on recommendation system solutions which could beat the baseline accuracy of the CineMatch algorithm used at the time. In this context, rating data was collected in a large sparse matrix filled with missing values in the entries associated to user-movie couples for which no rating was recorded, and a value from 1 to 5 corresponding to the evaluation a costumer assigned to a watched movie.

## The dataset

For the implementation, we downloaded the MovieLens 1M dataset downloaded at https://grouplens.org/datasets/movielens/. From the folder of interest, only the *rating.dat* file was exploited. It contains rating entries organized in 3 columns as <userID, movieID, rating>. In total, 1,000,209 ratings were recorded out of 6040 users and 3952 movies.

## The algorithms

The two algorithms we implemented hinge on matrix factorization. Given a sparse matrix $X$ with $u$ users on the rows and $m$ movies on the columns and an arbitrary constant $k$ such that $k << m,u$, the goal of the two implementations is to factorize $X$ to two smaller matrices $U$ and $M$ of size $u \times k$ and $k \times m$ such that the $u \times m$ matrix $P$ resulting from the dot product of $U$ and $M$ well approximates the non-missing entries of $X$. In other words, we seek to lower rank matrices which represent each user and movie 
