import streamlit as st
import pandas as pd
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import numpy as np



st.write("""
# Pulse Survey Results
Please note: sentiments may not be 100 percent accurate	""")



df = pd.read_csv('./df_cleaned_v1.csv')

#st.write(df['field1_Cleaned1'].head())


## Helper Functions
def generate_wordcloud(data, title, mask=None,colormap='RdYlGn'):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap=color_map,
                      mask=None,
                      background_color='white',
                      collocations=True).generate_from_text(data)
    plt.figure(figsize=(10,8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    st.pyplot()


### Sidebar Configuration
st.sidebar.write("Choose Filters")

f_sentiment = st.sidebar.selectbox('Filter by All or Negative Sentiment',('All', 'Negative'))
#Filter Dataframe based on sentiment

if f_sentiment=="All":
	df_select = df
else:
	df_select = df[df['Sentiment'].isin(['n'])]



##
## 2nd Filter : More and Better
st.sidebar.write("Keywords- More/Better")

f_keyword = st.sidebar.selectbox('Filter by More or Better Keywords',('All', 'More','Better', 'More and Better'))
if f_keyword=="All":
	df_select2 = df_select
	doc_string = "All Reviews"
elif f_keyword=="More":
	df_select2 = df_select[df_select['More']==1]
	doc_string = "Reviews with the keyword MORE"
elif f_keyword=="Better":
	df_select2 = df_select[df_select['better']==1]
	doc_string = "Reviews with the keyword BETTER"
else:
	df_select2 = df_select[(df_select['More']==1) | (df_select['better']==1)]
	doc_string = "Reviews with either of the keyword MORE or BETTER"


st.write(doc_string)
st.write(df_select2.head(20))



## Wordcloud
word_cloud_data = df_select2['field1_Cleaned1']
long_string = ','.join(list(word_cloud_data.values))
long_string=long_string.replace('nan', '')

#st.write(long_string)
color_map = st.selectbox(
'ColorMap',
('RdYlGn','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'))



generate_wordcloud(long_string, 'WordCloud', mask=None)
