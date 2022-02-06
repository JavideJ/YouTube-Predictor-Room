import re
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from googleapiclient.discovery import build


def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text) #Para eliminar las etiquetas HTML
    text = re.sub(r'[\W]+', ' ', text.lower()) #Eliminamos todos los caracteres que no sean palabras

    return text


def tokenizer(text):
    return nltk.word_tokenize(text,"spanish")


def tokenizer_porter(text):

    porter = PorterStemmer()
    return [porter.stem(word) for word in tokenizer(text)]

def preprocessor2(text):

    stop = nltk.corpus.stopwords.words("spanish")
    return " ".join([w for w in tokenizer_porter(text) if w not in stop])


def dur_total(duration):

    
    if ('H' in duration) & ('M' in duration) & ('S' in duration):
        
        list_dur_1 = duration.strip('PT').strip('S').split('H')
        list_dur_2 = list_dur_1[1].split('M')
        
        return int(list_dur_1[0])*3600 + int(list_dur_2[0])*60 + int(list_dur_2[1])
        
    elif ('H' in duration) & ('M' in duration):
        
        list_dur = duration.strip('PT').strip('M').split('H')
        
        return (int(list_dur[0])*3600 + int(list_dur[1])*60)
                         
    elif ('H' in duration) & ('S' in duration):
                         
        list_dur = duration.strip('PT').strip('S').split('H')
        
        return int(list_dur[0])*3600 + int(list_dur[1])
        
    elif 'H' in duration:
        
        return int(a.strip('PT').strip('H'))*3600
        
    elif ('M' in duration) & ('S' in duration):
        
        list_dur = duration.strip('PT').strip('S').split('M')
        return int(list_dur[0])*60 + int(list_dur[1])

    elif 'M' in duration:
        
        return int(duration.strip('PT').strip('M'))*60

    elif 'S' in duration:
        
        return int(duration.strip('PT').strip('S'))
    
    
def prediction(url, dislikes, value_):

    api_key = os.environ['api_key']     #key de la cuenta elCantrinho
    youtube = build('youtube', 'v3', developerKey=api_key) 
    

    try:

        yt_from_web = url.split('www.')[1][:7]

        if yt_from_web == 'youtube':

            url_id = url.split('v=')[-1].split('&t=')[0]

    except:
        pass

    try:

        yt_from_cell = url.split(r'://')[1][:8]

        if  yt_from_cell == 'youtu.be':

            url_id = url.split('.be/')[-1] 

    except:
        pass

    request_video = youtube.videos().list(part =[ 'topicDetails ', 'statistics', 'snippet','contentDetails'], id = url_id)                      
    response_video = request_video.execute()

    #Sacamos los datos del vídeo
    channel = response_video['items'][0]['snippet']['channelTitle']

    if channel == 'Rubius Z':
        channel = 'elrubiusOMG'

    title = response_video['items'][0]['snippet']['title']
    date = response_video['items'][0]['snippet']['publishedAt'].split('T')[0]
    time_delta = (datetime.now() - datetime.strptime(response_video['items'][0]['snippet']['publishedAt'].replace('T', ' ').strip('Z'), '%Y-%m-%d %H:%M:%S'))
    hours_on_air = round(time_delta.days * 24 + time_delta.seconds/3600, 2)
    views = int(response_video['items'][0]['statistics']['viewCount']) 
    likes = int(response_video['items'][0]['statistics']['likeCount'])
    duration = response_video['items'][0]['contentDetails']['duration']
    duration = dur_total(duration)
    comments = int(response_video['items'][0]['statistics']['commentCount'])

    dict_data = {'channel': channel, 'hours_on_air': hours_on_air, 'views': views, 'likes': likes, 'dislikes': dislikes, 'comments': comments, 'duration (sec)': duration}
    df= pd.DataFrame(dict_data, index=[0])

    #Limpiamos el título
    title = preprocessor2(preprocessor(title))

    with open(r'TfidfVectorizer.pickle', 'rb') as handle:
            tfidf = pickle.load(handle)

    title_tfidf = tfidf.transform(np.array([title]))

    df2 =pd.concat(objs = (pd.DataFrame(title_tfidf.toarray()), df),
                              axis = 1, ignore_index = True)

    df2.rename({9660:'channel', 9661: 'hours_on_air', 9662:'views', 9663: 'likes',	9664: 'dislikes', 9665: 'comments',	9666:'duration (sec)' }, 
              axis = 1, inplace = True)


    #Importamos los pickle del preprocesamiento
    with open(r'encoder_channel.pickle', 'rb') as handle:
        encoder_channel = pickle.load(handle)

    with open(r'scaler_without_channel.pickle', 'rb') as handle:
        scaler_without_channel = pickle.load(handle)


    #Scaler for numeric data
    data = scaler_without_channel.transform(df[df.drop(['channel'], axis = 1).columns])

    df2[df2[['hours_on_air', 'views', 'likes', 'dislikes', 'comments', 'duration (sec)']].columns] = data
    
    #This is to try later on this models on other youtubers
    channels_list = ['elrubiusOMG', 'Willyrex', 'aLexBY11', 'elxokas', 'JuegaGerman',
                    'Luisito Comunica', 'VEGETTA777', 'LOLiTO FDEZ', 'luzugames', 'TheGrefg']

    if value_ == ['1']:

        #Model importing
        with open(r'model_channel_NLP.pickle', 'rb') as handle:
            model1 = pickle.load(handle)

        y_hat = model1.predict(df2.drop(['channel'], axis = 1).values)
        
        return encoder_channel.inverse_transform(y_hat)[0], df.channel[0]

    
    elif value_ == ['2']:
        
        if df.channel[0] not in channels_list:
            with open(r'model_channel_NLP.pickle', 'rb') as handle:
                model1 = pickle.load(handle)
                
            df2['channel'] = encoder_channel.inverse_transform(model1.predict(df2.drop(['channel'], axis = 1).values))[0]
            
           
        #Label Encoder to the channels
        df2['channel'] = encoder_channel.transform(df2['channel'])
        
        #Model importing
        with open(r'model_views_NLP.pickle', 'rb') as handle:
            model2 = pickle.load(handle)
        
        df2['views'] = model2.predict(df2.drop(['views'], axis = 1).values)
        
        final_data = scaler_without_channel.inverse_transform(df2.drop(['channel'], axis = 1).iloc[:, -6:])
        
        return int(final_data[0][1]), views, round((int(final_data[0][1])-views)*100/views, 2)
        
    elif value_ == ['3']:
        
        if df.channel[0] not in channels_list:
            with open(r'model_channel_NLP.pickle', 'rb') as handle:
                model1 = pickle.load(handle)
                
            df2['channel'] = encoder_channel.inverse_transform(model1.predict(df2.drop(['channel'], axis = 1).values))[0]
           
        #Label Encoder to the channels
        df2['channel'] = encoder_channel.transform(df2['channel'])
        
        #Model importing
        with open(r'model_likes_NLP.pickle', 'rb') as handle:
            model3 = pickle.load(handle)
        
        df2['likes'] = model3.predict(df2.drop(['likes'], axis = 1).values)
        
        final_data = scaler_without_channel.inverse_transform(df2.drop(['channel'], axis = 1).iloc[:, -6:])
        
        return int(final_data[0][2]), likes, round((int(final_data[0][2]) - likes) *100 /likes, 2)
