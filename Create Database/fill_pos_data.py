# -*- coding: utf-8 -*-
"""
Fills the POS movements in the Postgres Database
@author: BM387
"""

import pandas as pd
import re
import psycopg2
from config import config

def extract_company_names(df, column):
    """
    Extracts the company name out of the description name. This method is a simple way of 
    extracting name by accessing the part after C/O or BEI 
    """
    entities = []
    for descr in df[column]:
        descr = str(descr)
        descr = re.sub(' +', ' ',descr)
        if " C/O " in descr:
            entity = descr.split(" C/O ")[1].strip()
        elif " BEI " in descr:
            entity = descr.split(" BEI ")[1].strip()
        else:
            entity = descr
        entities.append(entity)
        
    return entities



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



def assign_spending_cluster(df_movements_2018_POS, df_company_mapping):
    """
    Assigns spending cluster to each POS movement based on the company name
    """
    
    company_names = extract_company_names(df_movements_2018_POS, "DESCRIZIONE")
    df_movements_2018_POS["company name"] = company_names
    
    company_name_cluster = dict(zip(df_company_mapping["Seller name"],
                                df_company_mapping["CLUSTER"]))
    
    def get_cluster(company_name):
        """Simple lookup to get company cluster from manually assigned spending 
        type derived from company name"""
        if company_name in company_name_cluster:
            return company_name_cluster[company_name].upper() + "(POS)"
        else:
            return "UNASSIGNED" + "(POS)"


    df_movements_2018_POS["Cluster"] = df_movements_2018_POS["company name"].apply(get_cluster)

    return df_movements_2018_POS


def add_pos_rows(df):
    """
    Given the POS dataframe, add each row into the database
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
        sql = """ INSERT INTO pos (ndg_codificato, pos_date, description, spending_cluster, amount) VALUES (%s,%s,%s,%s,%s)"""
     
    
        for row in df.iterrows():
            
            date = str(row[1]["D_VAL2_VALC"])
            if  len(date)< 8:
                date = "0" + date
            
            data = (row[1]["NDG_CODIFICATO"], 
                    get_date(date), 
                    row[1]["DESCRIZIONE"], 
                    row[1]["Cluster"], 
                    -row[1]["E_IMP2_IMPC"])
            cur.execute(sql, data)
            
            nrows += 1
            
            if nrows % 1000000 == 0:
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
        

## Reading economic POS cluster assigned to each POS company name
df_company_mapping = pd.read_excel("./data fixed/company_cluster.xlsx")


## Loads POS movements for the year 2017, 2018 and 2019 in the postgres database
for year in [2017, 2018, 2019]:
    
    print("Reading  {} data".format(year))
    df_movements = pd.read_csv("./data {}/movements.csv".format(year), 
                                        sep = ";")
     
    # Extracting POS movements from the movement files only
    df_movements_pos = df_movements[(df_movements["DESCR_CAUS"] == "PAGAMENTO POS") 
                                    | (df_movements["DESCR_CAUS"] == "Pagamento POS")]


    # Assigning the POS cluster to each POS movement
    df_movements_pos  = assign_spending_cluster(df_movements_pos, df_company_mapping)

    # Inserting data into the database
    print("inserting  {} data".format(year))
    add_pos_rows(df_movements_pos)
    print("Done inserting  {} data".format(year))
    