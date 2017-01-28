from ZODB import FileStorage, DB
storage = FileStorage.FileStorage('mydatabase.fs')
db = DB(storage)
connection = db.open()
root = connection.root()
