import pandas as pd
from datetime import datetime
import pymorphy2

month_rus = {
    'январь': 1,
    'февраль': 2,
    'март': 3,
    'апрель': 4,
    'май': 5,
    'июнь': 6,
    'июль': 7,
    'август': 8,
    'сентябрь': 9,
    'октябрь': 10,
    'ноябрь': 11,
    'декабрь': 12
}

def filter_num(str):
    new_str = ''
    ascii_numbers = range(48, 58)
    for symbol in str:
        if ord(symbol) in ascii_numbers:
            new_str += symbol 

    try:
        num = int(new_str)
    except Exception:
        num = 1

    return num

def date_diff(str):
    str = str.lower()
    if 'мин' in str:
        return filter_num(str)
    elif 'час' in str:
        return filter_num(str)*60
    elif 'дней' in str or 'день' in str:
        return filter_num(str)*1440
    else:
        res = str.split(' ')
        res.reverse()
        try:
            res[1], res[2] = month_rus[res[2]], res[1][0:-1]
        except Exception:
            res = ['2023', month_rus[res[1]], res[0]]

        today = datetime.today()
        day = datetime(int(res[0]), int(res[1]), int(res[2]))
        return int((today - day).total_seconds()/60)

def filter_text(str):
    str = str.lower()
    new_str = ''
    ascii_numbers = list(range(1072, 1106))
    ascii_numbers.append(32)

    for symbol in str:
        if ord(symbol) in ascii_numbers:
            new_str += symbol 
        else:
            new_str += ' '

    return new_str


morth = pymorphy2.MorphAnalyzer()
functors_pos = {'INTJ', 'PRCL', 'CONJ', 'PREP'}

def drop_superfuluos(text):
    text = text.split()
    new_text = ''

    for word in text:
        if morth.parse(word)[0].tag.POS not in functors_pos:
            new_text += morth.parse(word)[0].normal_form
            new_text += ' '

    return new_text

def split_text(text):
    return text.split()

def create_words_embs(w2v_model, frame):
    indxs = [str(i) for i in range(0, 300)] #300 is a word vector size in the w2v model
    words_embs = pd.DataFrame(columns=[['word'] + indxs])

    for i, col in enumerate(frame.columns):
        try:
            emb = w2v_model.wv[col].tolist()
            words_embs.loc[i] = [col] + emb
        except Exception:
            continue
        
    words_embs = words_embs.reset_index().drop(columns=['index'])
    return words_embs

def w2v_tfidf_keys(w2v, tf_idf_frame):
    keys = []
    for col in tf_idf_frame.columns:
        try:
            w2v.wv[col]
            keys.append(col)
        except Exception:
            continue
    return keys