# streamlit application

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import streamlit as st
import pickle

st.title('Movie sentiment across models')

#Global variables
maxTokens = 20000 # max vocabulary size
max_length=600 # max document size
batch_size=32
# to be run at "C:/Users/plermant/git/miniprojects/NLPpractice"
nlpRoot="."

trainPath=nlpRoot+"/aclImdb/train"

@st.cache_resource
def getTrainDS(path, batch_size=batch_size):
    return keras.utils.text_dataset_from_directory(path, batch_size=batch_size)
    
@st.cache_resource 
def getTextTrainDS(_ds):
    return _ds.map(lambda x, y: x)
    
train_ds = getTrainDS(trainPath, batch_size=batch_size)
text_only_train_ds = getTextTrainDS(train_ds)

fileName=nlpRoot+'/models'+'/vectorTfIdf.pkl'
from_disk = pickle.load(open(fileName, "rb"))
text_vectorizationTfIdf = TextVectorization.from_config(from_disk['config'])
text_vectorizationTfIdf.set_weights(from_disk['weights'])


fileName=nlpRoot+'/models'+'/vectorLSTM.pkl'
from_disk = pickle.load(open(fileName, "rb"))
text_vectorizationBidiLSTM = TextVectorization.from_config(from_disk['config'])
text_vectorizationBidiLSTM.set_weights(from_disk['weights'])

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
  
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
        

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
  
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
 
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
 
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

@st.cache_resource
def modelFromName(name):
    path=nlpRoot+"/models/"+name
    if name=='bagBigramTdIdf' or name =='embedBidiLSTM':
        return tf.keras.models.load_model(path)	
    elif name == "transformer":
        return tf.keras.models.load_model(path, custom_objects={"TransformerEncoder": TransformerEncoder,"PositionalEmbedding": PositionalEmbedding})
    else:
        print("Error in model name, does not exist, aborting ...")
        exit(1)
 
@st.cache_resource 
def predictSentiment(inString,modelName):
    tfString=tf.convert_to_tensor([[inString],])
    model=modelFromName(modelName)
    inputs = keras.Input(shape=(1,), dtype="string")
    if modelName=='bagBigramTdIdf':
        processed_inputs = text_vectorizationTfIdf(inputs)
    elif modelName=='embedBidiLSTM' or modelName == 'transformer':
        processed_inputs = text_vectorizationBidiLSTM(inputs)
    outputs = model(processed_inputs)
    inference_model = keras.Model(inputs, outputs)
    predictions = inference_model(tfString) 
    resp=round(float(predictions[0] * 100),2)
    return resp
	
modelName= st.selectbox("Select a ML model from this list",("bagBigramTdIdf", 'embedBidiLSTM',"transformer"))
inputString= st.text_input('What do you think of the movie? e.g. "I loved the movie", or "This movie sucked"')

sentimentPercent=predictSentiment(inputString,modelName)
st.write("Sentiment value is",sentimentPercent,"per",modelName,"model.")
