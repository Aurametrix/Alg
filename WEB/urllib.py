# python -c "exec(urlopen(...).read())"
 
#Used to make requests
import urllib.request

x = urllib.request.urlopen('https://www.google.com/')
print(x.read())
