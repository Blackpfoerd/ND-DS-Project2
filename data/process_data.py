import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Description: Loads messages and categories files (csv) and merges them into a single DataFrame
    
    Input:
        messages_filepath:      string  --> path of messages (csv)
        categories_filepath:    string  --> path of categories (csv)  
    Output:
        pandas.DateFrame Object    
    """
    
    return df


def clean_data(df):
    """ 
    Description: Loads messages and categories files (csv) and merges them into a single DataFrame
    
    Input:
        messages_filepath:      string  --> path of messages (csv)
        categories_filepath:    string  --> path of categories (csv)  
    Output:
        pandas.DateFrame Object    
    """


def save_data(df, database_filename):
    """ 
    Description: Loads messages and categories files (csv) and merges them into a single DataFrame
    
    Input:
        messages_filepath:      string  --> path of messages (csv)
        categories_filepath:    string  --> path of categories (csv)  
    Output:
        pandas.DateFrame Object    
    """


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
