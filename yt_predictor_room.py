from flask import Flask, render_template, request
from NLP_prepro import prediction
import pymongo
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/services', methods = ['POST'])
def services():
    if request.method == 'POST':
        
        global url, dislikes
        url = request.form['url']
        dislikes = int(request.form['dislikes'])
        
        mongo_client = os.environ['mongo_client']
        db = pymongo.MongoClient(mongo_client)
        db_yt_url = db.yt_predictor.url_dislikes
        
        mongo_dict = {'url': url, 'dislikes': dislikes}
        result = db_yt_url.insert_one(mongo_dict)
              
        return render_template('service.html')

@app.route('/services/result', methods = ['GET', 'POST'])
def result():
    
    if request.method == 'POST':
        
        global value_
        value_ = request.form.getlist('mycheckbox')
        
        mongo_client = os.environ['mongo_client']
        db = pymongo.MongoClient(mongo_client)
        db_yt_url = db.yt_predictor.url_dislikes
        
        url = [k['url'] for k in db_yt_url.find()][-1]
        dislikes = [k['dislikes'] for k in db_yt_url.find()][-1]
    
        
        if value_ == ['1']:
            
            pred, real = prediction(url, dislikes, value_)
            
            return render_template('channel.html', pred = pred, real = real)
        
        elif value_ == ['2']:
            
            pred, real, error = prediction(url, dislikes, value_)
            
            return render_template('views.html', pred = pred, real = real, error = error)
            
        elif value_ == ['3']:
            
            pred, real, error = prediction(url, dislikes, value_)
            
            return render_template('likes.html', pred = pred, real = real, error = error)
        
        else:
            return 'I told you mark just one asshole'
    


if __name__ == '__main__':
    app.run(threaded=True, port = 3000, debug = False)
    
