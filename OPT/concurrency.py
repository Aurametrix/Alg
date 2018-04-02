from concurrent.futures import ThreadPoolExecutor
import time

import requests

def fetch(a):
    url = 'http://httpbin.org/get?a={0}'.format(a)
    r = requests.get(url)
    result = r.json()['args']
    return result

start = time.time()

# if max_workers is None or not given, it will default to the number of processors, multiplied by 5
with ThreadPoolExecutor(max_workers=None) as executor:
    for result in executor.map(fetch, range(30)):
        print('response: {0}'.format(result))

print('time: {0}'.format(time.time() - start))
