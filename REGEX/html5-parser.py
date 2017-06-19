from html5_parser import parse
from lxml.etree import tostring
root = parse(some_html)
print(tostring(root))
