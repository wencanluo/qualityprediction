import nltk
import random
import os
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import pickle, json
import fio

class QualityPredictionTrainer:
    def __init__(self, trainining_file, learning_algorithm = 'SVM'):
        '''
        learning_algorithm -- the algorithm to train with
            (default "SVM")
        '''
        
        self.trainining_file = trainining_file
        self.learning_algorithm = learning_algorithm
        
        self._model = None
        
        featuresets = self._get_training_data()
        self._train_classifier_model(featuresets)
    
    def get_features(self, text, cid, lecture):
        features = {}
        
        #unigram
        tokens = nltk.word_tokenize(text)
        
        features['WC'] = len(tokens)
        for token in tokens:
            features['U0'+token.lower()] = 1
        
        return features
        
    def get_model(self):
        """An accessor method for the model."""
        return self._model
    
    def _get_featuresets(self, input):
        featuresets = []
        
        MPLectures = fio.LoadDictJson(input)
        
        for week, MPs in enumerate(MPLectures):
            if MPs == []: continue
            
            for k, (MP, score) in enumerate(MPs):
                features = self.get_features(MP, week, 'Engineer')
                featuresets.append((features,score))
        
        return featuresets
        
    def _get_training_data(self):
        """Builds and returns positive and negative feature sets
        for the algorithm

        """
        featuresets = self._get_featuresets(self.trainining_file)
        return featuresets
    
    def _train_classifier_model(self, featuresets):
        """This changes the algorithm that nltk uses to train the model.

        Arguments:
        featuresets -- array of features generated for training

        """
        model = None
        print self.learning_algorithm
        if(self.learning_algorithm == "NB"):
            model = nltk.NaiveBayesClassifier.train(featuresets)
        elif(self.learning_algorithm == "MaxEnt"):
            model = nltk.MaxentClassifier.train(featuresets, "MEGAM",
                                                 max_iter=15)
        elif(self.learning_algorithm == "DecisionTree"):
            model = nltk.DecisionTreeClassifier.train(featuresets, 0.05)
        elif(self.learning_algorithm == 'SVM'):
            model = SklearnClassifier(LinearSVC()).train(featuresets)
        self._model = model

if __name__ == "__main__":
    classifier = QualityPredictionTrainer('../data/train.json', 'SVM')
    
    with open('../data/classifier_SVM.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)
    