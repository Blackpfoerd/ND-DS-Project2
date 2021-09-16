import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Description: Loads messages and categories files (csv) and merges them into a single DataFrame
    
    Input:
        messages_filepath: string = path of messages (csv)
        categories_filepath: string = path of categories (csv)  
    Output:
        pandas.DateFrame Object    
    """
    df_m = pd.read_csv(messages_filepath)
    df_c = pd.read_csv(categories_filepath)
    return pd.merge(df_m,df_c,how='inner',on='id')


def clean_data(df):
    """ 
    Description: Loads messages and categories files (csv) and merges them into a single DataFrame
    
    Input:
        df: pandas.DataFrame = to clean DataFrame 
    Output:
        pandas.DateFrame Object    
    """
    categories = df.categories.str.split(pat=';',expand=True)
    categories.columns = categories.iloc[1].str.split(pat='-').apply(lambda x: x[0])
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column],downcast='integer')
    df=df.drop(columns='categories').join(categories)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """ 
    Description: Saves the transformed DataFrame to database_filename.db
    table DisasterResponse
    
    Input:
        df: pandas.DataFrame = DataFrame 
        database_filename: string  --> path of saved database 
    Output:
        No output  
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse',engine,if_exists = 'replace', index=False)


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
