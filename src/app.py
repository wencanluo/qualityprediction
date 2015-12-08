#!flask/bin/python

from flask import Flask
app = Flask(__name__)

from flask import jsonify, request
from flask import make_response, abort

from QualityPrediction import QualityPrediction
from QualityPredictionTrainer import QualityPredictionTrainer 
import pickle

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/qualityprediction', methods=['GET'])
def usage():
    usage = 'This is the server to predict the reflection quality \n \
            Command line usage:\n \
            $curl -i -H "Content-Type: application/json" -X POST -d \'{"course":"IE256", "lecture":5, "text":"put a student response here", }\' http://coursemirror.cloudapp.net/qualityprediction'
    
    return jsonify({'usage': usage})

import random
@app.route('/qualityprediction', methods=['POST'])
def predict():
    if not request.json or not 'text' in request.json:
        abort(400)
    
    text = request.json['text']
    lecture = request.json['lecture'] if 'lecture' in request.json else None
    cid = request.json['course'] if 'course' in request.json else None
    
    score = qp.predict(text, cid, lecture)
    
    return jsonify({'course':cid,
                    'lecture':lecture,
                    'text':text,
                    'score': score})

if __name__ == "__main__":
    with open('../data/classifier_SVM.pickle', 'rb') as handle:
        classifier = pickle.load(handle)
        qp = QualityPrediction(classifier)
    
    app.run(host='0.0.0.0', port=80)
    