import pymysql
import os
import configparser

class DBConnectIni(object):
    def __init__(self,data):
        self.data=data

    def connection(url):
        CONFIG_FILE = url
        config = configparser.ConfigParser()
        # config = configparser.SafeConfigParser()
        config.read(CONFIG_FILE)
        # 第一个参数指定要读取的段名，第二个是要读取的选项名
        host = config.get("info","host")
        port = config.get("info","port")
        dbname = config.get("info","dbname")
        user = config.get("info", "user")
        password = config.get("info","password")

        db = pymysql.connect(host, user, password, dbname, charset="utf8")
        # 使用 cursor() 方法创建一个游标对象 cursor
        return db
