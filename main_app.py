import streamlit as st
import numpy as np
import pickle
import gensim
import gensim.downloader as api
import spacy
import string
from tensorflow.keras.models import load_model
import tensorflow as tf
import re
from spacy.lang.en.stop_words import STOP_WORDS
import time


# Load gensim model for embeddings
@st.cache_resource
def load_glove_vectors():
    glove_vectors_300 = api.load('glove-wiki-gigaword-300')
    return glove_vectors_300

# Preprocessing function
def preprocess(sent):
    '''Cleans text data up, leaving only 2 or
        more char long non-stopwords composed of A-Z & a-z only
        in lowercase'''
    # lowercase
    sentence = sent.lower()

    # Remove RT
    sentence = re.sub('RT @\w+: '," ",sentence)

    # Remove special characters
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", sentence)

    # Removing digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence) 
    # When we remove apostrophe from the word "Mark's", 
    # the apostrophe is replaced by an empty space. 
    # Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  
    # Next, we remove all the single characters and replace it by a space 
    # which creates multiple spaces in our text. 
    # Finally, we remove the multiple spaces from our text as well.

    return sentence


SPACY_MODEL_NAMES = "en_core_web_sm"

# Load spacy pipeline
# @st.cache(allow_output_mutation=True,persist=True,show_spinner=False)
@st.cache_resource
def load_spacy_model(name):
    nlp = spacy.load(name, exclude=["tagger", "ner",'attribute_ruler'])
    return nlp

nlp = load_spacy_model(SPACY_MODEL_NAMES)

# Tokenised text
# @st.cache(persist=True,show_spinner=False,suppress_st_warning=True,hash_funcs={"preshed.maps.PreshMap": hash,
# "cymem.cymem.Pool": hash,"thinc.model.Model": hash,"spacy.pipeline.tok2vec.Tok2VecListener": hash})
@st.cache_resource
def spacy_tokeniser(sentence):
    sentence  = sentence.strip().lower()
    doc = nlp(sentence)
    my_tokens = [word.lemma_ for word in doc if word.text not in STOP_WORDS]
    return my_tokens


# Vectoriser of generating embeddings
def sent_vec(sent, vectorizer):
    vector_size = vectorizer.vector_size
    w2v_resolution = np.zeros(vector_size)
    # print(w2v_resolution)
    ctr = 1
    for w in sent:
        if w in vectorizer:
            ctr += 1
            w2v_resolution += vectorizer[w]
    w2v_resolution = w2v_resolution/ctr
    # print(w2v_resolution)
    return w2v_resolution


raw_example_text = """
The keys keep failing to click when tapped/punched. I hate this product. 
WHy THE HECK DID I PURCHASE IT!!!!!!!!!!
"""
st.header(":red[This app classifies your text as] :white[mouse], :green[headphones] or :blue[keyboard_speakers]")

txt = st.text_input("Text to classify",
                    raw_example_text,
                    placeholder="Awaiting your own text")



@st.cache_resource
def load_models():
    models  = {
        "mlp":pickle.load(open('models/mlp_word_embeddings.pkl', "rb")),
        "svm":pickle.load(open('models/svm_word_embeddings.pkl', "rb")),
        "bilstm":load_model('models/biLSTM-model.keras')
    }

    return models


models = load_models()


classes = {0:'mouse', 1:'headphone', 2:'keyboard_speakers'}
classifier_choice = st.selectbox(
    'Select a classifier',
    ['mlp','svm', 'bilstm'],
    index=0)

def prediction_mapper(predictions,probabilities,classes):
    
    # Flatten the predictions into 1D array
    predictions = np.array(predictions).flatten()
    probabilities = np.array(probabilities)
    
    # Extracting the scalar value of predictions
    for i,pred_index in enumerate(predictions):
        predicted_class = classes[pred_index]
        probs = np.max(probabilities[i]) if probabilities.ndim > 1 else np.max(probabilities)
        st.success(f"The review is talking about {predicted_class}, probability is {round(np.max(probs),4)}")


glove_embeddings = load_glove_vectors()

if txt:
    # Calling all predefined functions
    processed_text = preprocess(txt)
    lemmatised_text = spacy_tokeniser(processed_text)
    if classifier_choice in ['mlp','svm']:
        classifier = models[classifier_choice]
        X = sent_vec(lemmatised_text,glove_embeddings)

        # Reshaped vector for prediction
        X_reshaped = [X]

        # Predictions 
        y_pred = classifier.predict(X_reshaped)
        if hasattr(classifier,"predict_proba"):
            y_pred_proba = classifier.predict_proba(X_reshaped)
        else:
           decision_scores = classifier.decision_function(X_reshaped) 
           # Getting an approximate probability from decision function of SVM
           min_scores  = decision_scores.min(axis=1,keepdims=True)
           max_scores  = decision_scores.max(axis=1,keepdims=True)
           y_pred_proba = (decision_scores - min_scores)/ (max_scores- min_scores)


        my_bar  = st.progress(0)
        for percentage_completion in range(100):
            time.sleep(0.01)
            if percentage_completion == 50:
                st.write("Almost done predicting your text!")
            my_bar.progress(percentage_completion +1 )
        
        time.sleep(0.01)
        st.markdown("Prediction done! 😁")
        st.balloons()
        time.sleep(0.01)
        prediction_mapper(y_pred,y_pred_proba,classes)
        

    elif classifier_choice == 'bilstm':
        loaded_tokenizer = pickle.load(open("models/tokeniser.pkl",'rb'))
        classifier = models[classifier_choice]
        
        # Convert text to sequences
        X_seq = loaded_tokenizer.texts_to_sequences([lemmatised_text])
        max_len = 285
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_seq,
                                                          padding='post',
                                                          maxlen=max_len)
        # Make predictions
        y_pred_proba = classifier.predict(X_padded)
        y_pred = np.argmax(y_pred_proba,axis=1)

        my_bar  = st.progress(0)
        for percentage_completion in range(100):
            time.sleep(0.01)
            if percentage_completion == 50:
                st.write("Almost done predicting your text!")
            my_bar.progress(percentage_completion +1 )

        time.sleep(0.01)
        st.markdown("Prediction done! 😁")
        st.balloons()
        time.sleep(0.01)
        prediction_mapper(y_pred,y_pred_proba,classes)