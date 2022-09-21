from sklearn.linear_model import LogisticRegression

class Model:
    """
    Base class
    """
    
    def train(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_pred):
        raise NotImplementedError
        

class Logistic_regression(Model):
    
    def __init__(self):
        self.classifier = LogisticRegression(penalty='l2', max_iter=3000)
    
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_pred):
        return self.classifier.predict(X_pred)
        
        