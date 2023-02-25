from sklearn.feature_extraction.text import TfidfVectorizer

#once
'''import nltk
nltk.download("stopwords")'''

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from preprocessing import *

data = pd.read_csv('../FotoCaptions/data/data.csv')

data = data[data['likes'] != 'другим']
data = data.dropna()

data['likes'] = data['likes'].apply(filter_num)
data['time'] = data['time'].apply(date_diff)
data = data.reset_index().drop(columns=['index'])

data = data[data['likes'] < 5000]
data = data[data['time'] <= data['time'].mean()]

data['likes_per_day'] = ((1440 * data['likes']) / (data['time'])).astype(int)
data = data[data['likes_per_day'] < 25000]

data['popularity'] = (data['likes_per_day'] > data['likes_per_day'].quantile(q=0.75)).astype(int)
data = data.reset_index().drop(columns=['index'])

data['text'] = data['text'].apply(filter_text)
data['text'] = data['text'].apply(drop_superfuluos)


stopwords = stopwords.words("russian")
tf_idf_vec = TfidfVectorizer(use_idf=True,  
                        smooth_idf=True,  
                        ngram_range=(1,1),stop_words=stopwords)
tf_idf_data = tf_idf_vec.fit_transform(data['text']) 
tf_idf_frame = pd.DataFrame(tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names_out())


data['text'] = data['text'].apply(split_text)

w2v = Word2Vec(min_count=20, window=4, negative=10, vector_size=300, alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=1)
w2v.build_vocab(data['text'])
w2v.train(data['text'], total_examples=w2v.corpus_count, epochs=30, report_delay=1)

keys = w2v_tfidf_keys(w2v, tf_idf_frame)
tf_idf_frame = tf_idf_frame[keys]

new_data = pd.concat([tf_idf_frame, data['popularity']], axis=1)
new_data.info()
new_data.to_csv('../FotoCaptions/data/prepared_data.csv', index=False)