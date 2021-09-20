# Disaster Response Pipeline Project

## Motivation:
Every day countless messages are posted to social media platforms like Twitter, Instagram, Facebook & more. A small share of this messages include requests for help, food and shelter. In this project I want to help finding this messages with a data analytics solution.

## Project Summary
In this project I will set up a ML Service that will help to classify messages from social media platforms to identify people in need. Therefore I will train a Multiclass Classifier based on two datasets provided by Figure Eight.

## Instructions:
### Required libraries
You will need to install following packages:
- nltk 
- numpy
- pandas
- scikit-learn
- sqlalchemy

### Structure

- `app/`
  - `template/`
    - `master.html`  #Main page of web application.
    - `go.html`  #Classification result page of web application.
  - `run.py`  #Flask applications main file.

- `data/`
  - `disaster_categories.csv`  #Disaster categories dataset.
  - `disaster_messages.csv`  #Disaster Messages dataset.
  - `process_data.py` #ETL script
  - `DisasterResponse.db` #Database including cleaned data

- `models/`
  - `train_classifier.py` #ML pipeline
  - `train_classifier.pkl` #Pickled model



#Files
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgements
Thanks to Figure Eight for open dataset.

