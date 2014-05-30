needle = raw_input("Enter a string: ")

# old-fashioned comprehension

if needle.endswith('ly') or needle.endswith('ed') or\
    needle.endswith('ing') or needle.endswith('ers'):
    print('It is valid')
else:
    print('It is invalid')

# more elegant comprehension

if any([needle.endswith(e) for e in ('ly', 'ed', 'ing', 'ers')]):
    print('It is valid')
else:
    print('It is invalid')
    
# more pythonic way
    
from re import search as re_search
# importing re.search as re_search to decrease the overhead for the lookup

def old_fashioned(needle):
    return bool(needle.endswith('ly') or needle.endswith('ed') or\
        needle.endswith('ing') or needle.endswith('ers'))

def list_comprehension(needle):
    return bool(any([needle.endswith(e) for e in ('ly', 'ed', 'ing', 'ers')]))
        
def generator(needle):
    return bool(any(needle.endswith(e) for e in ('ly', 'ed', 'ing', 'ers')))

def endswith_tuple(needle):
    return bool(needle.endswith(('ly', 'ed', 'ing', 'ers')))

def regexpr(needle):
     return bool(re_search(r'(?:ly|ed|ing|ers)$', needle))
    
def map_func(needle):
    return any(map(needle.endswith, ('ly', 'ed', 'ing', 'ers')))
    
import re

#faster
#re_search(r'(?:ly|ed|ing|ers)$', 'needlers')
# slower
#re.search(r'(?:ly|ed|ing|ers)$', 'needlers')

#funcs = [old_fashioned, list_comprehension, 
#         generator, endswith_tuple, regexpr,
#         map_func
#        ]