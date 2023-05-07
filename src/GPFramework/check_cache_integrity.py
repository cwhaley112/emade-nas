import sqlalchemy
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import time
import random
import os

username = ''
password = ''
server = ''
database = ''
connection_string = 'mysql+pymysql://' + username + ':' + password + '@' + server + '/' + database

dataset = "datasets/spambase"

if __name__ == '__main__':
    my_engine = sqlalchemy.create_engine(connection_string)
    session = sessionmaker(bind=my_engine)()
    
    session.execute("START TRANSACTION;")
    rows = session.execute("SELECT id, train_directory, test_directory FROM cache WHERE dirty=0;")
    print("Clean Rows Count: ", rows.rowcount)
    
    for row in rows:
        if not os.path.exists(row.train_directory + row.id) or not os.path.exists(row.test_directory + row.id):
            print(row.train_directory + row.id)
            print(row.test_directory + row.id)
            print("Cache Integrity Failed")
        
    session.execute("ROLLBACK;")
    
    session.execute("START TRANSACTION;")
    rows = session.execute("SELECT id, train_directory, test_directory FROM cache WHERE dirty=1 or dirty=2;")
    print("Dirty Rows Count: ", rows.rowcount)
    
    for row in rows:
        if os.path.exists(row.train_directory + row.id) or os.path.exists(row.test_directory + row.id):
            print(row.train_directory + row.id)
            print(row.test_directory + row.id)
            print("Cache Integrity Failed")
        
    session.execute("ROLLBACK;")
    
    error_1_count = 0
    error_2_count = 0
    
    folds = [f.path for f in os.scandir(dataset) if f.is_dir()]    
    for fold in folds:
        entries = [f.path for f in os.scandir(fold) if f.is_dir()]  
        for entry in entries:
            c_hash = entry[len(fold)+1:]
            if len(c_hash) > 9:
                
                session.execute("START TRANSACTION;")
                row = session.execute("SELECT * FROM cache WHERE id='{}'".format(c_hash)).first()
                if row is None:
                    print("Missing a row which should be there. Hash:", c_hash)
                    print("Entry: ", entry)
                    print("Cache Integrity Failed")
                    error_1_count += 1
                elif row.dirty == 1 or row.dirty == 2:
                    print("Data stored which should NOT be there. Hash:", c_hash)
                    print("Dirty:", row.dirty)
                    print("Entry: ", entry)
                    print("Cache Integrity Failed")
                    error_2_count += 1
                session.execute("ROLLBACK;")
    
    print("Done")
    
    print("Error 1 Count:", error_1_count)
    print("Error 2 Count:", error_2_count)
    
    session.close()