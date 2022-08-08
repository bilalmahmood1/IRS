# -*- coding: utf-8 -*-
"""
Fill the controlling information into the postgres database
@author: BM387
"""


import pandas as pd
import psycopg2
from config import config


def add_controlling_rows(df):
    """
    Given the controlling dataframe, add each row into the database
    """    
    conn = None
    try:
        # read database configuration
        params = config
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        sql = """ INSERT INTO controlling (ndg_codificato, controlling_date, debiti, crediti, indiretta) VALUES (%s,%s,%s,%s,%s)"""
     
    
        for row in df.iterrows():
            data = (row[1]["ndg_codificato"], row[1]["date"], row[1]["DEBITI"], row[1]["CREDITI"], row[1]["INDIRETTA"])
            cur.execute(sql, data)
       
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()            
                
    except (Exception, psycopg2.DatabaseError) as error:
           print(error)
    finally:
        if conn is not None:
            conn.close()
            
            
def create_date(mese , year):
    """
    Creates date in European format YYYY-MM-DD (2017-03-30)
    """
    return str(int(year)) + "-" + str(int(mese)) + "-1"


# Filling the controlling information for 2017, 2018 and 2019 into the database
for year in [2017, 2018, 2019]:

    # Reading the controlling data for the specific year
    df = pd.read_csv("./data {}/controlling.csv".format(year),sep = ";")
    
    print("inserting  {} data".format(year))
    df.fillna(0, inplace = True)
    
    # Creating date
    df["date"] = df.apply(lambda row : create_date(row["mese"], 
                                                   row["anno"]), 
                          axis = 1) 

    # Inserting the controlling dataframe rows into the database 
    add_controlling_rows(df)
    
    print("Done inserting  {} data".format(year))

