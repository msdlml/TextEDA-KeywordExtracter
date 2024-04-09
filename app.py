from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
#import threading
import torch


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

model = SentenceTransformer('xlm-r-large-en-ko-nli-ststb')
keybert_model = KeyBERT(model)

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text)
    text = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.String(100))

@app.route('/', methods=['GET', 'POST'])
def index():
    with app.app_context():
        db.create_all()
        keywords = None
        if request.method == 'POST':
            title = request.form['title']
            text = request.form['text']
            torch.cuda.empty_cache()
            # 키워드 추출
            okt = Okt()
            nouns = okt.nouns(text)
            nouns_text = ' '.join(nouns)
            keywords = keybert_model.extract_keywords(nouns_text, keyphrase_ngram_range=(1, 1), stop_words=None)
            keywords_data = ' '.join([keyword[0] for keyword in keywords])
            new_data = Keyword(title = title, text=text, keywords=keywords_data)
            db.session.add(new_data)
            db.session.commit()
        db_data = Keyword.query.order_by(Keyword.id.desc()).all()
            
        if db_data:
            return render_template('index.html', keywords=keywords, db_data=db_data)
        else:
            return render_template('index.html', keywords=keywords)
        #return render_template('index.html', keywords=keywords, db_data = db_data)

#def collect_and_store_data():
    #while True:
        
        # 데이터 크롤링 혹은 수집 로직 작성 후 text 변수에 담았다고 가정함
        #title = "수집된 제목"
        #text = "수집된 본문"

        # 키워드 추출
        # okt = Okt()
        # nouns = okt.nouns(text)
        # nouns_text = ' '.join(nouns)
        # model = SentenceTransformer('xlm-r-large-en-ko-nli-ststb')
        # keybert_model = KeyBERT(model=model)
        # keywords = keybert_model.extract_keywords(nouns_text, keyphrase_ngram_range=(1, 1), stop_words=None)

        #new_data = Keyword(title = title, text=text, keywords=keywords)
        #db.session.add(new_data)
        #db.session.commit()


#data_collection_thread = threading.Thread(target=collect_and_store_data)
#data_collection_thread.start()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    app.run(debug=True)