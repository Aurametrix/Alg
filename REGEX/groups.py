import re

#   ^ matches the beginning of a string.
#	$ matches the end of a string.
#	\b matches a word boundary.
#	\d matches any numeric digit.
#	\D matches any non-numeric character.
#	(x|y|z) matches exactly one of x, y or z.
#	(x) in general is a remembered group. We can get the value of what matched by using the groups() method of the object returned by re.search.
#	x? matches an optional x character (in other words, it matches an x zero or one times).
#	x* matches x zero or more times.
#	x+ matches x one or more times.
#	x{m,n} matches an x character at least m times, but not more than n times.

def getpatterns(str):
  match = re.search('([\w.-]+)@([\w.-]+)', str)
  if match:
    print match.group()   ## 'alice-b@google.com' (the whole match)
    print match.group(1)  ## 'alice-b' (the username, group 1)
    print match.group(2)  ## 'google.com' (the host, group 2)

#without round brackets there are no groups
  match = re.search('[\w.-]+@[\w.]+', str)
  if match:
    print "simplified", match.group()   ## the whole match

#The 'r' at the start of the pattern string designates a python "raw" string which passes 
# through backslashes without change which is very handy for regular expressions
#match = re.search('(\d{3})\D*(\d{3])\D*(\d{4})', str)
#match = re.search(r'([\d+])-([d{3}])', str)
  match = re.search(r'(\d{3})-(\d{3})-(\d{4})', str)
  if match:
    print "phone number in your string", match.group()    
    
  str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
  umails = re.findall(r'([\w\.-]+)@([\w\.-]+)', str)
  print umails  ## [('alice', 'google.com'), ('bob', 'abc.com')]
  for um in umails:
    print um[0]  ## username
    print um[1]  ## host
        
import sys 
def main():
  str = 'purple alice-b@google.com monkeydishwasher 305-786-3222 irene@mail.com 111-11-1111'
  str = raw_input("Enter a string with phone numbers and e-mails:  ")
  #if (len(sys.argv) > 2):
  #      str = map(int,sys.argv[1:])        # Get string from input, convert to int
  print "here's what I've got:", str 
  output = getpatterns(str)
  print "here's the result", output
 
if __name__ == '__main__':
    main()
