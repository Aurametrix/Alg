#!/usr/bin/python
# match checks the beginning of the line unlike search and perl regex

import re
import sys

if (len(sys.argv) < 2):
    print "please tyoe a sentence and I will match it to my templates" 
        
#line = ' '.join(str(sys.argv[1:]));
line = ' '.join(sys.argv[1:]);

#line = "Cats are smarter than dogs";

print line;

matchObj = re.match( r'dogs', line, re.M|re.I)
if matchObj:
   print "match --> matchObj.group() : ", matchObj.group()
else:
   print "Matched - No match!!"

matchObj = re.search( r'dogs', line, re.M|re.I)
if matchObj:
   print "search --> matchObj.group() : ", matchObj.group()
else:
   print "Searched - No match!!"

line = raw_input("Type a sentence   ")
print line

#line = " dogs and cats"
lookfor = "dogs";
print line.find(lookfor);

if (line.find(lookfor,0,len(line))>-1):
   print "Found: ", lookfor
else:
   print "No match!!"

