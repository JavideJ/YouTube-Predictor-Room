from flask import Flask, render_template, request
from NLP_prepro import prediction

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
        
        print(url)
              
        return render_template('service.html')

@app.route('/services/result', methods = ['GET', 'POST'])
def result():
    
    if request.method == 'POST':
        
        global value_
        value_ = request.form.getlist('mycheckbox')
    
        
        if value_ == ['1']:
            print(url)
            
            pred, real = prediction(url, dislikes, value_)
            
            return render_template('channel.html', pred = pred, real = real)
        
        elif value_ == ['2']:
            
            pred, real, error = prediction(url, dislikes, value_)
            
            return render_template('views.html', pred = pred, real = real, error = error)
            
        elif value_ == ['3']:
            
            pred, real, error = prediction(url, dislikes, value_)
            
            return render_template('likes.html', pred = pred, real = real, error = error)
        
        else:
            return 'I told you mark just one :)'
    


if __name__ == '__main__':
    app.run(threaded=True, port = 3000, debug = False)
    
