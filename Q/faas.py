{
 "url": "https://blog.alexellis.io/rss/",
 "term": "docker"
}

# update the hello-python/handler.py file

import requests  
import json

def handle(req):  
    result = {"found": False}
    json_req = json.loads(req)
    r = requests.get(json_req["url"])
    if json_req["term"] in r.text:
        result = {"found": True}

    print json.dumps(result)
