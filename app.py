import streamlit as st
import pandas as pd
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import numpy as np



st.write("""
# Pulse Survey Results
Please note: sentiments may not be 100 percent accurate	""")



df = pd.read_csv('./df_cleaned_v1.csv')

st.write(df['field1_Cleaned1'].head())


## Helper Functions
def generate_wordcloud(data, title, mask=None):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap='RdYlGn',
                      mask=None,
                      background_color='white',
                      
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()


### Sidebar Configuration
st.sidebar.write("Choose Filters")

f_sentiment = st.sidebar.selectbox('Filter by All or Negative Sentiment',('All', 'Negative'))

st.write(f_sentiment)

## Wordcloud
word_cloud_data = df['field1_Cleaned1']
long_string = ','.join(list(word_cloud_data.values))
long_string=long_string.replace('nan', '')


generate_wordcloud(long_string, 'WordCloud', mask=None)
