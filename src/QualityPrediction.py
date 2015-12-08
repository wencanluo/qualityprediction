from QualityPredictionTrainer import QualityPredictionTrainer 
import pickle

class QualityPrediction():
    def __init__(self, model):
        self._model_object = model
    
    def predict(self, text, cid=None, lecture=None):
        features = self._model_object.get_features(text, cid, lecture)
        return self._model_object.get_model().classify(features)
    

if __name__ == "__main__":
    with open('../data/classifier_SVM.pickle', 'rb') as handle:
        classifier = pickle.load(handle)
        qp = QualityPrediction(classifier)
    
    print qp.predict('Nothing')
    print qp.predict('Van der waal bonding')
        