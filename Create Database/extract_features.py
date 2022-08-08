# -*- coding: utf-8 -*-
"""
SQL Helper functions to query databases to extract relevant features in producing customer profiles
@author: BM387
"""

from config import *
import psycopg2
import pandas as pd


def get_max_gain_spending_sql(ndg_codificato, start_date, end_date):
    """
    Returns maximum and minimum month over month change in wealth of the 
    customer(debiti + indiretta)
    """    
    return """
        
    SELECT abs(min(increase)) diminiure, abs(max(increase)) aumentare FROM
    (
    	SELECT *, (debiti + indiretta) as wealth,
    	(debiti + indiretta) - lag((debiti + indiretta)) over (order by controlling_date) as increase
    	from controlling 
    	where ndg_codificato = {0} AND
    	'{1}' <= controlling_date AND controlling_date <= '{2}' 
    	ORDER BY controlling_date
    ) as wealth_diff
    
    """.format(ndg_codificato, start_date, end_date)
    
    
def get_controlling_sql(ndg_codificato, start_date, end_date, months):
    """
    Returns the SQL to get the Controlling pattern of the individual
    """
    
    return """
    SELECT sum(debiti)/{0} as average_balance, sum(indiretta)/{0} as average_stock, sum(debiti + indiretta)/{0} average_wealth FROM controlling 
    where ndg_codificato = {1} AND
    '{2}' <= controlling_date AND controlling_date <= '{3}' 
    """.format(months, ndg_codificato, start_date, end_date)
    

def get_spending_sql(ndg_codificato, start_date, end_date, months):
    """
    Returns the SQL to get the POS spending pattern of the individual
    """
        
    return """
    SELECT A.spending_cluster, B.ms / {}
    FROM
    (SELECT * FROM spending_cluster) as A
    LEFT JOIN 
    (SELECT 
    monthly_spending.spending_cluster,
    sum(monthly_spending.spending) as ms
    FROM
    (
    	SELECT spending_cluster, to_char(pos_date, 'YYYY-MM') as month_year, sum(amount) as spending
    	FROM pos where ndg_codificato = {} AND
    	'{}' <= pos_date AND pos_date <= '{}' GROUP BY to_char(pos_date, 'YYYY-MM'), spending_cluster  ORDER BY month_year DESC 
    ) as monthly_spending
    GROUP BY 
    monthly_spending.spending_cluster
    ) AS B
    ON A.spending_cluster = B.spending_cluster
    """.format(months, ndg_codificato, start_date, end_date)


def get_salary_sql(ndg_codificato, start_date, end_date, months):
    """
    Returns the SQL statement to get salary of the individual
    """
    return """
    
    SELECT -sum(amount) / {}
    from movement
    where ndg_codificato = {} AND '{}' <= movement_date  AND movement_date <= '{}'	
    """.format(months, ndg_codificato, start_date, end_date)




def get_gain_spending_information(ndg_codificato, start_date, end_date):
    """ 
    Query pos table to get the maximum gain and maximum spending statistics 
    for a particular user
    """
    conn = None
    try:
        params = config
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        sql = get_max_gain_spending_sql(ndg_codificato, start_date, end_date)
        print(sql)
        cur.execute(sql)
        row = cur.fetchone()
        result = []
        for r in row:
            if pd.isna(r):        
                result.append(0)        
            else:
                result.append(r)
                         
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally: 
        if conn is not None:
            conn.close()
            
        return result


def get_spending_information(ndg_codificato, start_date, end_date, months):
    """ Query pos table to get the spending statistics 
    for a particular user
    """
    conn = None
    try:
        params = config
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        sql = get_spending_sql(ndg_codificato, start_date, end_date, months)
        print(sql)
        cur.execute(sql)
        row = cur.fetchall()
        
        df_spending = pd.DataFrame(row)
        df_spending.fillna(0,inplace = True)
        
        df_spending = df_spending.set_index(0)
        print(df_spending)
        
                         
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally: 
        if conn is not None:
            conn.close()
            
        return df_spending
    


def get_salary_information(ndg_codificato, start_date, end_date, months):
    """ Query movements table to get the salary statistics 
    for a particular user
    """
    conn = None
    try:
        params = config
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        sql = get_salary_sql(ndg_codificato, start_date, end_date, months)
        print(sql)
        cur.execute(sql)
        row = cur.fetchone()
        result = []
        for r in row:
            if pd.isna(r):        
                result.append(0)        
            else:
                result.append(r)
                         
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally: 
        if conn is not None:
            conn.close()
            
        return result
    


def get_controlling_information(ndg_codificato, start_date, end_date, months):
    """ Query controlling table to get the controlling statistics 
    for a particular user
    """
    conn = None
    try:
        params = config
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        sql = get_controlling_sql(ndg_codificato, start_date, end_date, months)
        print(sql)
        cur.execute(sql)
        row = cur.fetchone()
        result = []
        for r in row:
            if pd.isna(r):        
                result.append(0)        
            else:
                result.append(r)
                         
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally: 
        if conn is not None:
            conn.close()
            
        return result
    
  
def get_pos_spending_materialized_view():
    """
    View creation for creating POS clusters
    """
    return """
    CREATE MATERIALIZED VIEW spending_cluster
    AS
    (
        SELECT spending_cluster
        FROM pos
        GROUP By 
        spending_cluster  
    )
    """