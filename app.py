import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy
import pandas
import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords


p = PorterStemmer()

app = Flask(__name__)

## Load the model
model = pickle.load(open("model.pkl","rb"))

vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data  = str(request.form.values())
    print(data)

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y =[]
        for i in text:
            if i.isalnum():
                y.append(i)
            
        text = y[:]
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
            
        text = y[:]
        y.clear()
    
        for i in text:
            y.append(p.stem(i))
 
        return " ".join(y)

    transform_data = transform_text(data)

    vectors = vectorizer.transform([transform_data])
    ans = model.predict(vectors)
    print(ans)
    if ans == 0:
        return render_template("home.html",prediction_text="Not Spam")
    else:
        return render_template("home.html",prediction_text=" Spam")
    # if ans==0:
    #     return "Not Spam"
    # else:
    #     return "Spam"

@app.route("/predict",methods=['POST'])
def predict():
    data = [request.form.values()]
    print(data)
    





if __name__ =="__main__":
    app.run(debug=True)