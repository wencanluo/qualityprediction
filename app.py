#!flask/bin/python

from flask import Flask
app = Flask(__name__)

from flask import jsonify, request
from flask import make_response, abort

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
    if not request.json or not 'text' in request.json or not 'lecture' in request.json or not 'course' in request.json:
        abort(400)
    
    score = random.randint(0,3)
    
    return jsonify({'course':request.json['course'],
                    'lecture':request.json['lecture'],
                    'text':request.json['text'],
                    'score': score})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    