import streamlit as st
import pandas as pd
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


@st.cache
def load_data():

    df = pd.read_csv("augmented_datafile.csv")
    lb = LabelEncoder()
    lb.fit(df['Categories'])
    df['Target'] = lb.transform(df['Categories'])
    df['Target'] = lb.transform(df['Categories'])
    return df,lb
df,lb = load_data()

@st.cache
def load_models():
    model = tf.keras.models.load_model("final_model.h5",custom_objects={'KerasLayer': hub.KerasLayer})
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    kw_model_bert = KeyBERT(model=sbert_model)
    summarizer = pipeline("summarization") 
    return kw_model_bert,model, summarizer 
kw_model_bert,model,summarizer = load_models()

 
def main():

    st.set_page_config(
        page_title="AWDClassifier",
        page_icon="🎈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.markdown("# AWDClassification")
    st.sidebar.text("\n\n\n")
    option = 'Home Page'
    option = st.sidebar.selectbox('NLP Service',('Home Page','Text Topic', 'Keywords Extraction', 'Text Summarization'))


    if option == 'Home Page':
        st.image("Text_Classification_image.jpg", width=700)
        st.markdown("### Welcome To The Multi-class Text Classification Web Application ")
        st.write("In the graph below, you can see a brief description of the data by showing the number of elements in each class.")
        st.image("data_plot.png",width=800)
        st.markdown("\n What type of NLP service would you like to use? you can choose one of the options in the sidebar")


    elif option=='Text Topic':

        st.subheader("Enter the text you'd like to analyze.")
        input = st.text_area('Enter text and press Ctrl+Enter to apply') #text is stored in this variable

        
        if (input!=""):
            #st.success("#### The main topic of the inserted text is 'Artificial Intelligence'")

            key_txt = kw_model_bert.extract_keywords(input, keyphrase_ngram_range=(1, 2), nr_candidates=15, top_n=10,stop_words='english')
            k = str(list(dict(key_txt).keys()))
            prediction = model.predict([tf.convert_to_tensor([input]) ,tf.convert_to_tensor([k]) ])
            result = lb.inverse_transform(np.argmax(prediction).reshape(1)) # Get the class name
            st.success("#### The main topic of the inserted text is'",result[0],"'")

            satisfaction = st.slider('Tell us how satisfied you are with the result?', 0, 100, 25)
            st.write("\n If you are not satisfied with the result, please help our model to better distinguish the context by filling in the box below with the correct result.")
            st.text_input(label="Please insert the correct result")
            
            st.button("Submit")

        else :
            st.error("Please enter your Text")




    elif option == 'Keywords Extraction':
        st.subheader("Enter the text you want to analyze and get the most relevant keywords with their score compared to 1.")
        input2 = st.text_area('Enter text') #text is stored in this variable
        key = kw_model_bert.extract_keywords(input2, keyphrase_ngram_range=(1, 2), nr_candidates=15, top_n=10,stop_words='english')
        keys = list(dict(key).keys())
        st.sucess(str(keys))
        

        
    
    else:
        st.subheader("Enter the text you'd like to analyze.")
        input3 = st.text_area('Enter text') #text is stored in this variable
        st.subheader("The Summary")
        if(input3 !=""):

            result = summarizer(input3, min_length=20, max_length=70, do_sample=False)
            summary = result[0]["summary_text"]
            st.sucess(summary)


        #st.success("Animal law is a combination of statutory and case law in which the nature legal social or biological of nonhuman animals is an important factor . Animal law permeates and affects most traditional areas of the law including tort contract criminal and constitutional law . Growing number of state and local bar associations now have animal law committees .")





main()