import streamlit as st
import pandas as pd
import numpy as np 
from wordcloud import WordCloud
import matplotlib.pyplot as plt




# loading in all the essentials for data manipulation
import pandas as pd
import numpy as np
#load inthe NTLK stopwords to remove articles, preposition and other words that are not actionable
from nltk.corpus import stopwords
# This allows to create individual objects from a bog of words
from nltk.tokenize import word_tokenize
# Lemmatizer helps to reduce words to the base form
from nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
from nltk import ngrams
# We can use counter to count the objects
from collections import Counter
# This is our visual library
import seaborn as sns
import matplotlib.pyplot as plt






import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Pulse Survey Results
Please note: sentiments may not be 100 percent accurate   
Hover mouse on the row to read the entire review.
	""")

stopwords = ['more','better','a','the','to','and','for','of','that','are','i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']

df = pd.read_csv('./df_cleaned_v1.csv')


df.drop(['field1'], axis=1, inplace=True)
#st.write(df['field1_Cleaned1'].head())


## Helper Functions
def generate_wordcloud(data, title, mask=None,colormap='RdYlGn'):
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap=color_map,
                      mask=None,
                      background_color='white',
                      stopwords=stopwords,
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
st.write(df_select2['field1_Cleaned1'].head(50))



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




def word_frequency(sentence):
	# joins all the sentenses
	#sentence =" ".join(sentence)
	# creates tokens, creates lower class, removes numbers and lemmatizes the words
	new_tokens = word_tokenize(sentence)
	new_tokens = [t.lower() for t in new_tokens]
	new_tokens =[t for t in new_tokens if t not in stopwords]
	new_tokens = [t for t in new_tokens if t.isalpha()]
	lemmatizer = WordNetLemmatizer()
	new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
	#counts the words, pairs and trigrams
	counted = Counter(new_tokens)
	counted_2= Counter(ngrams(new_tokens,2))
	counted_3= Counter(ngrams(new_tokens,3))
	#creates 3 data frames and returns thems
	word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
	word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
	trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
	return word_freq,word_pairs,trigrams


word_freq,word_pairs,trigrams = word_frequency(long_string)


word_freq_20 = word_freq[0:19]
word_pairs_20 = word_pairs[0:19]
trigrams_20 = trigrams[0:19]


## Plot 

fig, axes = plt.subplots(3,1,figsize=(8,20))
sns.barplot(ax=axes[0],x='frequency',y='word',data=word_freq_20)
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=word_pairs_20)
sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=trigrams_20)
st.pyplot(fig)
