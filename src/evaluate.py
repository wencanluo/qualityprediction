from QualityPrediction import QualityPrediction
import pickle

if __name__ == '__main__':
    
    import ConfigParser
    import sys
    
    config = ConfigParser.RawConfigParser()
    config.read('../config/default.cfg')
    
    for feature in [#'WC', 
                    #'unigram', 
                    #'pos',
                    #'WC,pos',
                    #'WC,pos,unigram',
                    'WC,pos,unigram,content',
                    'WC,pos,unigram,organization',
                    'WC,pos,unigram,content,organization',
                    ]:
        config.set('model','features',feature)
        
        model = QualityPrediction(config)
        
        metric = model.evaluate()
        
        print feature, metric
    