import json
from elasticsearch import Elasticsearch

es = Elasticsearch()

# by default we connect to localhost:9200
# from elasticsearch import E1

# create an index in elasticsearch, ignore status code 400 (index already exists)
 es.indices.create(index='my-index', ignore=400)
{u'acknowledged': True}

resp = E1.search(index="mynewcontacts", body={"query": {"match_all": {}}})
    response = json.dumps(resp)
    data = json(load)
    #print data["hits"]["hits"][0]["_source"]["email"]
    for row in data['hits']['hits']:
    print row["hits"]["hits"][0]["_source"]["email"]
    return "OK"
    
PUT my_index/my_type/1
{
  "text": "Geo-point as an object",
  "location": { 
    "lat": 41.12,
    "lon": -71.34
  }
}
