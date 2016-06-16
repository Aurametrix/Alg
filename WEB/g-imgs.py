import urllib2
import json as simplejson
import io as cStringIO

fetcher = urllib2.build_opener()
searchTerm = 'Aurametrix'
startIndex = 0
searchUrl = "http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=" + searchTerm + "&start=" + startIndex
f = fetcher.open(searchUrl)
searchUrl = ('https://ajax.googleapis.com/ajax/services/search/images?' +
       'v=1.0&q=barack%20obama&userip=INSERT-USER-IP')
deserialized_output = simplejson.load(f)
