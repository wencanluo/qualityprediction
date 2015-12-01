#!flask/bin/python

from flask import Flask
app = Flask(__name__)

from flask import jsonify, request
from flask import make_response, abort

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

import random
@app.route('/qualityprediction', methods=['POST'])
def predict():
    if not request.json or not 'text' in request.json:
        abort(400)
    
    score = random.randint(0,3)
    
    return jsonify({'text':request.json['text'],
                    'score': score})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    