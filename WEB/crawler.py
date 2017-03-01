from .tasks import longtime_add
import time
if __name__ == '__main__':
    url = ['http://example1.com' , 'http://example2.com' , 'http://example3.com' , 'http://example4.com' , 'http://example5.com' , 'http://example6.com' , 'http://example7.com' , 'http://example8.com'] # change them to your ur list.
    for i in url:
        result = longtime_add.delay(i)
        print 'Task result:',result.result
        
from __future__ import absolute_import
from test_celery.celery import app
import time,requests
from pymongo import MongoClient
client = MongoClient('10.1.1.234', 27018) # change the ip and port to your mongo database's
db = client.mongodb_test
collection = db.celery_test
post = db.test
@app.task(bind=True,default_retry_delay=10) # set a retry delay, 10 equal to 10s
def longtime_add(self,i):
    print 'long time task begins'
    try:
        r = requests.get(i)
        post.insert({'status':r.status_code,"creat_time":time.time()}) # store status code and current time to mongodb
        print 'long time task finished'
    except Exception as exc:
        raise self.retry(exc=exc)
    return r.status_code
    
  version: '2'
services:
    rabbit:
        hostname: rabbit
        image: rabbitmq:latest
        environment:
            - RABBITMQ_DEFAULT_USER=admin
            - RABBITMQ_DEFAULT_PASS=mypass
        ports:
            - "5673:5672"

    worker:
        build:
            context: .
            dockerfile: dockerfile
        volumes:
            - .:/app
        links:
            - rabbit
        depends_on:
            - rabbit
    database:
        hostname: mongo
        image: mongo:latest
        ports:
            - "27018:27017"
