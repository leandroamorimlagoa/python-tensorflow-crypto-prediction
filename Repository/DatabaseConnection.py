import mysql.connector
from mysql.connector import errorcode
from AppConfiguration import AppConfiguration



class DatabaseConnection:
    def __init__(self):
        try:
            __databaseConfig = AppConfiguration.get_instance().get_config()["DatabaseConnection"]
            self.conn = mysql.connector.connect(
                host=__databaseConfig["host"],
                user=__databaseConfig["user"],
                password=__databaseConfig["password"],
                database=__databaseConfig["database"]
            )
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Error: Access denied.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Error: Database not exists.")
            else:
                print("Error: ", err)

    def get_connection(self):
        return self.conn

    def close_connection(self):
        if self.conn:
            self.conn.close()

