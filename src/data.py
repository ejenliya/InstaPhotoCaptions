from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from preprocessing import *

data = pd.read_csv('../FotoCaptions/data/data.csv')

data = data[data['likes'] != 'другим']
data = data.dropna()

data['likes'] = data['likes'].apply(filter_num)
data['time'] = data['time'].apply(date_diff)
data = data.reset_index().drop(columns=['index'])

data['text'] = data['text'].apply(filter_text)
data['text'] = data['text'].apply(drop_superfuluos)


tf_idf_vec = TfidfVectorizer(use_idf=True,  
                        smooth_idf=True,  
                        ngram_range=(1,1),stop_words='english')
tf_idf_data = tf_idf_vec.fit_transform(data['text']) 
tf_idf_frame = pd.DataFrame(tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names_out())


data['text'] = data['text'].apply(split_text)

w2v = Word2Vec(min_count=15, window=4, negative=10, vector_size=300, alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=1)
w2v.build_vocab(data['text'])
w2v.train(data['text'], total_examples=w2v.corpus_count, epochs=30, report_delay=1)

words_embs = create_words_embs(w2v, tf_idf_frame)
words_embs.info()

keys = [el[0] for el in words_embs['word'].values]
tf_idf_frame = tf_idf_frame[keys]
tf_idf_frame.info()

data['likes_per_day'] = ((1440 * data['likes']) / (data['time'])).astype(int)
data['popularity'] = (data['likes_per_day'] > data['likes_per_day'].median()).astype(int)

new_data = pd.concat([tf_idf_frame, data['popularity']], axis=1)
new_data.head()
new_data.to_csv('../FotoCaptions/data/prepared_data.csv', index=False)