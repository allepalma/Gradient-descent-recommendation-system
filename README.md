# Recommendation systems

The present repository contains the python3 implementation of 2 different recommendation systems where sparse records of movie ratings are leveraged to produce predictions for the level of satisfaction a user is expected to report in relation to a movie he/she has not rated before. The algorithms are built upon the concept of collaborative filtering such that the prediction of a missing rating for a user is produced through a model trained on a large dataset of historical rating records.
The setting we adopted for the implementation was inspired by the challenge launched by Netflix in 2006 which prompted data scientists worldwide to work on recommendation system solutions which could beat the baseline accuracy of the CineMatch algorithm used at the time. In this context, rating data was collected in a large sparse matrix filled with missing values in the entries associated to user-movie couples for which no rating was recorded, and a value from 1 to 5 corresponding to the evaluation a costumer assigned to a watched movie.

## The d
