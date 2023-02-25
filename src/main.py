'''
This bot analyzes the texts in the captions of Instagram posts
and their impact on the popularity of publications.
'''

import telebot
from prediction import make_prediction
import yaml

with open('../FotoCaptions/config/config.yaml') as f:
    temps = yaml.safe_load(f)

TOKEN = temps['token']
LANGUAGE = None


bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    global LANGUAGE
    LANGUAGE = None
    bot.send_message(message.chat.id, f'Hello, {message.from_user.first_name}!\n' +
                     '/setlanguage you prefer ' +
                     'and write me the /text you want to place under the photo on Instagram, ' +
                     'and I will tell you how successful it is.')

@bot.message_handler(commands=['help'])
def help_command(message):
   markup = telebot.types.InlineKeyboardMarkup()
   markup.add(
       telebot.types.InlineKeyboardButton('Message the developer', url='telegram.me/ejenliya')
   )
   bot.send_message(
       message.chat.id,
       '/start to start and reset the bot\n\n' +
       '/setlanguage to select language you prefer\n' +
       '/text to write and check your photo caption\n',
       reply_markup=markup
   )

@bot.message_handler(commands=['setlanguage'])
def set_language(message):
    markup = telebot.types.InlineKeyboardMarkup()
    item1 = telebot.types.InlineKeyboardButton('RUS', callback_data='RUS')
    item2 = telebot.types.InlineKeyboardButton('EN', callback_data='EN')
    markup.add(item1, item2)
    if not LANGUAGE or LANGUAGE == 'EN':
        bot.send_message(message.chat.id, 'Choose the language you prefer to write caption in:', reply_markup=markup)
    elif LANGUAGE == 'RUS':
        bot.send_message(message.chat.id, 'Выбери язык, на котором хочешь написать текст:', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def callback(query):
   if query.message:
        global LANGUAGE
        LANGUAGE = query.data
        if LANGUAGE == 'RUS':
            bot.send_message(query.message.chat.id, 'Отлично!')
        elif LANGUAGE == 'EN':
            bot.send_message(query.message.chat.id, 'Sorry, this language is not supporting yet :(')

@bot.message_handler(commands=['text'])
def text(message):
    if LANGUAGE:
        if LANGUAGE == 'RUS':
            bot.send_message(message.chat.id, 'Напиши мне свой текст:')
            bot.register_next_step_handler(message, get_user_text_rus)
        elif LANGUAGE == 'EN':
            bot.send_message(message.chat.id, 'Sorry, this language is not supporting yet :(')
    else:
        bot.send_message(message.chat.id, 'Please select the language before!')
        set_language(message)

def get_user_text_rus(message):
    caption = message.text

    try:
        pred = make_prediction(caption)[0]
    except ValueError:
        bot.send_message(message.chat.id, 'Здесь нет слов. Дай мне слова!')
    else: 
        try:
            if pred[0] > 0.4 and pred[0] < 0.6:
                bot.send_message(message.chat.id, 'Не уверен...')
        except TypeError:
            bot.send_message(message.chat.id, 'Здесь нет слов, которые влияют на популярность. Попробуй ещё!')
        else:
            bot.send_message(message.chat.id, f'Пост наберет популярность с вероятностью {round(pred[1]*100, 2)}%. Попробуй ещё раз!')

bot.polling(non_stop=True)