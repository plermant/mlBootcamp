We compare 3 different NLP models to process movie review sentiment.
- Bag of bigram words with tf-idf
- Bi-di LSTM
- Transformer

We initially train each model with the aclImdb public dataset and save the 3 models to file

A browser-streamlit application asks the user to pick from the UI one of the 3 models and input a movie review string, like "i loved this movie".
The UI sends request to server, which returns a percentage of positive sentiment, that gets displayed in UI.

The streamlit app is dockerized and can be installed either locally or in the cloud.

First run the jupyter notebook to create all needed assets. Uncomment relevant cells to initially fetch the raw aclImd public dataset. 
Then create a docker container using the Dockerfile file.
Deploy the image on a server and run a docker continer
It will launch the sreamlit app and can be communicated with from a browser on port 8501
