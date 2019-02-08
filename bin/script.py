############
# Date : 2018.03.20
# Chungbuk Natioanl University AI lab
#########
import sqlite3
from pandas import DataFrame
import pandas as pd
import numpy as np
import os.path

database = 'db.sqlite3'

def createAndInsert(dir):
    dir = "media/" + str(dir)
    print(dir)

    data = pd.read_csv(dir)
    rowdata = data.head(10)  #출력 갯수 제한
    colCNT = (data.shape)  # (row, columns)
    CNT= colCNT[1] # 컬럼 수

    tableName = getTableName(dir)
    create_table(tableName, data, CNT)
    insert_table(tableName, rowdata, CNT)

def getTableName(dir):
    dir = str(dir)                  # 경로를 string타입으로 변환
    tablenm, tableext = os.path.splitext(dir) # 파일경로 및 확장 추출
    nm = tablenm.split("/")[-1]     # 테이블 이름
    return nm

def selectData_(tableName):
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create projects table
        cur = conn.cursor()

        # SQL 쿼리 실행
        query = "SELECT * FROM " + tableName
        cur.execute(query)

        # 데이타 Fetch
        rows = (cur.fetchall())

    else:
        print("Error! cannot create the database connection.")

    conn.close()

    return rows

#############################################
def selectData(tableName):
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        try:
            # create projects table
            cur = conn.cursor()

            # SQL 쿼리 실행
            query = "SELECT * FROM " + tableName
            cur.execute(query)

            # 데이타 Fetch
            rows = (cur.fetchall())

        except sqlite3.Error as e:
            print(e)
            conn.close()
            rows = None

    else:
        print("Error! cannot create the database connection.")

    return rows

#############################################
def create_table(tableName, data, CNT):
    dfield =[] #컬럼명

    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # 컬럼 생성
        for c in range(CNT):
            dfield.append(data.columns[c])
        ###################
        ## 테이블 생성
        ##################
        query = 'CREATE TABLE IF NOT EXISTS ' + tableName + ' ( '
        for c in range(CNT):
            if(c == CNT-1):
                query = query + dfield[c] + " text"
            else:
                query = query + dfield[c] + " text, "

        query = query + ');'

        try:
            c = conn.cursor()
            c.execute(query)
        # Exception
        except sqlite3.Error as e:
            print(e)
    else:
        print("Error! cannot create the database connection.")

    conn.close()

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file, timeout=60)
        return conn
    except sqlite3.Error as e:
        print("connect", e)
    return None

def insert_table(tableName, insertdata, CNT):
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        try:
            iris_list = []
            iris_list = pd.DataFrame(insertdata)
            c = conn.cursor()

            # prepare statement query문 생성
            query = "insert into " + tableName + " values ("

            for i in range(1, CNT):
                query += "?, "

            query += "?)"

            c.executemany(query, iris_list.values)
            conn.commit()

        # Exception
        except sqlite3.Error as e:
            print (e)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

def getColumnName(tableName):
    con = create_connection(database)
    cur = con.cursor()
    query = "SELECT * FROM " + tableName
    cur.execute(query)
    result = cur.fetchall()

    names = list(map(lambda x: x[0], cur.description))

    con.close()

    return names

######################## for test main
"""
#def main():
   # create a database connection
    conn = create_connection(database)
    if conn is not None:
        # create projects table
        create_table(conn)
        insert_table(conn, rowdata)

    else:
        print("Error! cannot create the database connection.")


    m = my_custom_sql("abc")

    conn.close()
"""

#############################################
def select_loss(epoch):
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        try:
            # create projects table
            cur = conn.cursor()

            # SQL 쿼리 실행
            query = "SELECT loss FROM " + "Loss" + " WHERE epoch = '" + str(epoch) + "'"
            print(query)
            cur.execute(query)

            # 데이타 Fetch
            rows = (cur.fetchall())

        except sqlite3.Error as e:
            print(e)
            conn.close()
            rows = None

    else:
        print("Error! cannot create the database connection.")

    return rows

#############################################
def select_accuracy(epoch):
    # create a database connection
    conn = create_connection(database)
    if conn is not None:
        try:
            # create projects table
            cur = conn.cursor()

            # SQL 쿼리 실행
            query = "SELECT accuracy FROM " + "Accuracy" + " WHERE epoch = '" + str(epoch) + "'"
            cur.execute(query)

            # 데이타 Fetch
            rows = (cur.fetchall())

        except sqlite3.Error as e:
            print(e)
            conn.close()
            rows = None

    else:
        print("Error! cannot create the database connection.")

    return rows


createAndInsert("mnist_1.csv")