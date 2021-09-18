import sys
import pandas as pd
import re
import pickle
import numpy as np

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(database_filepath):
    """ 
    Description: Loads data from database
    
    Input:
        database_filepath: string = path of database  
    Output:
        X: pandas.DataFrame = Subset of df with predictor Columns
        y: pandas.DataFrame = Subset of df with target columns
        category_name: list = list of categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df= pd.read_sql_table('DisasterResponse', engine)
    X= df.message
    y= df[df.columns.difference(['message','original','genre','id'])]
    
    category_names=y.columns
    return X,y,category_names

def tokenize(text):
    """ 
    Description: Tokenizer function for CountVectorizer
    
    Input:
        text: string =   
    Output:
        clean_tokens: list = returns list of cleaned tokens
    """  
    text = re.sub(r'[^a-z0-9]'," ", text.lower())
    #tokenize and remove stop words
    tokens = word_tokenize(text)
    #tokens_wo_stop= [word for word in tokens if word not in stopwords.words('english')]
    
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens =[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
   
    return clean_tokens


def build_model():
    """ 
    Description: Returns Pipeline Object
    
    Input:
        --  
    Output:
        model: Pipeline = Returns ML Pipeline
    """
    pipeline=Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', RandomForestClassifier())
        ('clf', MultiOutputClassifier(RandomForestClassifier(),n_jobs=1))
    ])
    #If you want other parameters get them with pipeline.get_params().keys()
    parameters = {
        'clf__estimator__max_depth': range(1,10,1)
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, verbose=1)
    
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Description: Evaluates model and prints out 
    
    Input:
        --  
    Output:
        --
    """
    
    Y_pred = model.predict(X_test)
    #class_rep = classification_report(Y_test,Y_pred, target_names=np.array(category_names))
    accuracy= (Y_pred == Y_test).mean()
    
    print("Label:", category_names)
    #print("Confusion Matrix\n", class_rep)
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    """ 
    Description: Saves model to model filepath
    
    Input:
        model: Object = Classifier
        model_filepath: string = location to save the model
    Output:
        --
    """
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
      
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
