import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import nltk.corpus
import re
import string
import pickle

#Reading the data
df = pd.read_csv(r'C:\Users\DELL\Desktop\imbd_movie\IMDB Dataset.csv')

#Turning sentiment into categorical values
# 1 - positive , 0 - negative
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
df['sentiment'] = l.fit_transform(df['sentiment'])

#Remove special characters from text
pattern = re.compile(r'<br\s*/><br\s*/>>*|(\-)|(\\)|(\/)')
def preprocess_reviews(reviews):
  reviews = [pattern.sub('',item) for item in reviews]
  return reviews
clean = preprocess_reviews(df['review'])

#Removing punctuations
pattern = re.compile(r'[^\w\s]')
def remove_punctuations(reviews):
  reviews = [ pattern.sub('',item) for item in reviews]
  return reviews
clean = remove_punctuations(df['review'])
df['review'] = clean

#Converting the text to lowercase
df['review'] = df['review'].str.lower()

#Removing line breaks
def remove_linebreaks(input):
  pattern = re.compile(r'\n')
  return pattern.sub('',input)
df['review'] = df['review'].apply(remove_linebreaks)

#Tokenization
from nltk.tokenize import word_tokenize
df['review'] = df['review'].apply(word_tokenize)

#Removing stopwords
from nltk.corpus import stopwords
def remove_stopwords(reviews):
 return [w for w in reviews if w not in stopwords.words('english')]
df['review'] = df['review'].apply(lambda x: remove_stopwords(x))

#Lemmatization
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
def word_lemmatize(input):
  return [lem.lemmatize(word) for word in input]
df['review'] = df['review'].apply(word_lemmatize)

#Combining all the individual words
def combine_words(input):
  combined = ' '.join(input)
  return combined
df['review'] = df['review'].apply(combine_words)

y = df.iloc[:,-1].values

from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer(min_df=2,max_df=0.5,ngram_range=(1,3))
X = tfid.fit(df['review'])
X = tfid.transform(df['review'])
X.toarray()

# Creating a pickle file for the TfidfVectorizer
pickle.dump(tfid, open('tfidf-transform.pkl', 'wb'))


#Splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Model Building
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'movie-reviews-sentiment-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
