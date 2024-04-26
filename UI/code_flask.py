from flask import Flask,render_template,request
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import time

warnings.filterwarnings('ignore')
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        start_time = time.time()
        df = pd.read_csv("data/train.csv")
        df.isnull().sum()

        Toxic_comment_df=df.loc[:,['id','comment_text','toxic']]
        Toxic_comment_balanced_1 = Toxic_comment_df[Toxic_comment_df['toxic'] == 1].iloc[0:5000,:]
        Toxic_comment_balanced_0 = Toxic_comment_df[Toxic_comment_df['toxic'] == 0].iloc[0:5000,:]

        Toxic_comment_balanced=pd.concat([Toxic_comment_balanced_1,Toxic_comment_balanced_0])
        train, test = train_test_split(Toxic_comment_balanced, random_state=42, test_size=0.30, shuffle=True)

        train_text = train['comment_text']
        test_text = test['comment_text']

        vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
        vectorizer.fit(train_text)
        vectorizer.fit(test_text)

        x_train = vectorizer.transform(train_text)
        y_train = train.drop(labels = ['id','comment_text'], axis=1)

        x_test = vectorizer.transform(test_text)
        y_test = test.drop(labels = ['id','comment_text'], axis=1)

        X = Toxic_comment_balanced.comment_text
        y = Toxic_comment_balanced['toxic']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initiate a Tfidf vectorizer
        tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

        X_train_fit = tfv.fit_transform(X_train)  
        X_test_fit = tfv.transform(X_test)  
        randomforest = RandomForestClassifier(n_estimators=100, random_state=50)

        randomforest.fit(X_train_fit, y_train)
        randomforest.predict(X_test_fit)

        comment1 = [request.form['toxic']]
        print(comment1)
        comment1_vect = tfv.transform(comment1)
        result = randomforest.predict_proba(comment1_vect)[:,1]
        result = result[0]*100
        message = ""
        msg = ""
        print(result)

        # for toxic percentage
        
        message = result
                
        # for toxic or not message
        if(result > 74):
            msg = "Entered comment is toxic"
            print('Entered comment is toxic')    
        else:
            msg = "Entered comment is not toxic"
            print('Entered comment is not toxic')
        end_time = time.time()
        execution_time = end_time - start_time

    return render_template('index.html',result = message, res = msg, execution_time= execution_time)

    

if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.debug = True
	app.run()
