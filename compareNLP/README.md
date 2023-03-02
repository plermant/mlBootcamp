We compare 3 different NLP models to process movie review sentiment.
- Bag of bigram words with tf-idf
- Bi-di LSTM
- Transformer

We initially train each model with the aclImdb public dataset and save models to file

A streamlit application asks user to pick from a browser UI one of the 3 models and a movie review string.
The UI sends request to server, which returns a percentage of positive sentiment, that gets displayed in UI.

The streamlit app is dockerized and can be installed either locally or in the cloud.

First run the jupyter notebook to create all needed assets. Uncomment relevant cells to initially fetch the raw aclImd public dataset
Create a file named streamlitApp.py by copying last cell from the Jupyter notebook 
Then create a docker container using the Dockerfile file.
Deploy the image on a server. It will launch the sreamlit app and can be communicated with from a browser on port 8501
