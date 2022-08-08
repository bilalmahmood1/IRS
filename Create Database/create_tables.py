# -*- coding: utf-8 -*-
"""

Create Sparkasse tables: Controlling, pos, and movement, that are essential for building profiles
@author: BM387
"""

import psycopg2
from config import config
        
def create_tables():
    """ create tables in the PostgreSQL database"""  
     
        
    commands = [
       """
       CREATE TABLE controlling (
            id SERIAL PRIMARY KEY,
            ndg_codificato Bigint NOT NULL,
            controlling_date DATE NOT NULL,
            debiti real,
            crediti real,
            indiretta real    
        )
        """,
        """
        CREATE TABLE pos (
            id SERIAL PRIMARY KEY,
            ndg_codificato Bigint NOT NULL,
            pos_date DATE NOT NULL,
            description TEXT,
            spending_cluster TEXT,
            amount real    
        )
        """,
          """
        CREATE TABLE movement (
            id SERIAL PRIMARY KEY,
            ndg_codificato Bigint NOT NULL,
            movement_date DATE NOT NULL,
            description TEXT,
            amount real    
        )
        """
        ]
    conn = None
    try:
        # read the connection parameters
        params = config
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
    
        for command in [commands[-1]]:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    create_tables()
    print("Done creating your tables!")
    