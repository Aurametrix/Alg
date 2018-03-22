from th2c import AsyncHTTP2Client
from tornado.httpclient import HTTPRequest

client = AsyncHTTP2Client(
    host='nghttp2.org', port=443, secure=True,
)

request = HTTPRequest(
    url='https://nghttp2.org/httpbin/get',
    method='GET',
)

res = yield client.fetch(request)
