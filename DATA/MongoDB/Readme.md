###
 
Drivers:
[PyMongo](https://github.com/mongodb/mongo-python-driver) 

The PyMongo distribution contains tools for interacting with MongoDB database from Python. 
+ The bson package is an implementation of the BSON format for Python. 
+ The pymongo package is a native Python driver for MongoDB. 
+ The gridfs package is a gridfs implementation on top of pymongo.



For other languages: https://docs.mongodb.com/ecosystem/drivers/

node.js [MongoDB driver](https://mongodb.github.io/node-mongodb-native/) and on [GitHub](https://github.com/christkv/node-mongodb-native)
[also](https://github.com/mafintosh/mongojs)

Several other tools that sit on top of this driver: Mongoose -  an ORM tool, but completely unnecessary.
Mongoskin or Mongolia provide less verbose access than the "native" driver.

Python + MongoDB - getting started
https://www.w3schools.com/python/python_mongodb_getstarted.asp

PyMongoQueries
https://www.w3schools.com/python/python_mongodb_query.asp


+ [MongoDB 3.4.0-rc3](https://jepsen.io/analyses/mongodb-3-4-0-rc3)
+ [4.2.6](https://jepsen.io/analyses/mongodb-4.2.6)

https://www.mongodb.com/download-center#community to Download MongoDB Community Server.

“Run service as Network Service user”
pip install pymongo


http://bsonspec.org/

Jepsen analysis of databases  http://jepsen.io/analyses



Read generic binary files:

    import struct
    with open("aurametrix_production.0", mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    struct.unpack("iiiii", fileContent[:20])

 read bson
 
    import bson
    with open('diary_entries.bson','rb') as f:
        data = bson.decode_all(f.read())
    print(data)  
