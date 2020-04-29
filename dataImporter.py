import sys

import pandas as pd
import codecs
import MySQLdb as db
import pymysql
from sqlalchemy import create_engine

def reformatFile(filename):
    with codecs.open(filename, 'r', 'cp1251') as f:
        lines = f.readlines()
    lines = [line.replace('\t', '') for line in lines]
    lines = [line.replace('\n', '') for line in lines]

    with codecs.open(filename, 'w', 'cp1251') as f:
        f.writelines(lines)

def insertToDB(dataFrame, table):
    engine = create_engine('mysql+pymysql://root@127.0.0.1:6606/qazaq_base?charset=utf8mb4')
    dataFrame.to_sql(table, con=engine, if_exists='append', index=False)

def importAbbreviations(filename):
    abbreviations = pd.read_table(filename, sep='-', names=('word', 'deciphered'), error_bad_lines=False)
    insertToDB(abbreviations, 'abbreviations')

if __name__ == "__main__":
    importAbbreviations('abbreviations.txt')
