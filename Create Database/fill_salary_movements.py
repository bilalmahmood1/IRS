# -*- coding: utf-8 -*-
"""

Fills the Salary movements in the Postgres Database

@author: BM387
"""

import pandas as pd
import re
import psycopg2
from config import config


def get_date(text):
    """
    Gets string date from the text
    """
    text = str(text)
    if len(text) < 8:
        text = "0" + text
    day = text[:2]
    month = text[2:4]
    year = text[4:]
    return year + "-" + month + "-" + day


def add_movements_rows(df):
    """
    Given the movements dataframe, without POS, dataframe, add each row into the database
    """
    conn = None
    nrows = 0
    try:
        # read database configuration
        params = config
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        sql = """ INSERT INTO movement (ndg_codificato, movement_date, description, amount) VALUES (%s,%s,%s,%s)"""
     
        for row in df.iterrows():
            
            date = str(row[1]["D_VAL2_VALC"])
            if  len(date)< 8:
                date = "0" + date
            
            data = (row[1]["NDG_CODIFICATO"], 
                    get_date(date), 
                    row[1]["DESCR_CAUS"], 
                    -row[1]["E_IMP2_IMPC"])
            cur.execute(sql, data)
            
            nrows += 1
            
            if nrows % 100000 == 0:
                conn.commit()
                
       
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()            
                
    except (Exception, psycopg2.DatabaseError) as error:
           print(error)
    finally:
        if conn is not None:
            conn.close()
            


# Filling salary movements for the years 2017, 2018 and 2019
for year in [2017, 2018, 2019]:
    
    print("Reading  {} data".format(year))
    df_movements = pd.read_csv("./data {}/movements.csv".format(year), 
                                        sep = ";")
    
    ## Extacting the salary movements
    df_salaries = df_movements[df_movements["DESCR_CAUS"] == 'EMOLUMENTI']
    
    
    ## Inserting the salary movements
    print("inserting  {} data".format(year))
    add_movements_rows(df_salaries)
    print("Done inserting  {} data".format(year))
    print("Done filling the salary")
    

