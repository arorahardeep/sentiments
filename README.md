# sentiments

This is a sentiments analysis example using LSTM. I tried 2 different datasets

a. News sentiments on Dow Jones Index - https://www.kaggle.com/aaron7sun/stocknews  
b. Amazon food review dataset - https://www.kaggle.com/snap/amazon-fine-food-reviews

stocknews dataset is really small and all attempts to learn on such small dataset don't generalize. 
The training accuracy get upto 99% but the validation accuracy does not go beyond 55%

amazon food review data set is huge with around 0.5m data points. I train on first 50k examples and use a stacked LSTM
network with Embedding layer. I downloaded Glove weights for the embedding layer.
This training accuracy get up to 95% and the validation accuracy around 93%. When I run it next 100k unseen examples
the accuracy drops to 82%.

## Files

The following module loads the stocknews dataset
> load_data.py

The following module loads first 50k of the amazon dataset, it expects embeddings/glove.6B.100d.txt and amazon/Reviews.csv
It considers 0 & 1 stars as negative reviews, removes 3 stars and considers 4 & 5 stars as positive reviews
> load_data_amz.py

The following module trains the LSTM for sentiments analysis, 
> sentiments.py

The following module loads the last saved module and retrains it for more epochs.
> retrain_sentiments.py

The following module loads the trained model and uses it to predict sentiments on a dataset. You would need to adjust the 
dataset on which to train in the load_data_amz.py
> predict.py
