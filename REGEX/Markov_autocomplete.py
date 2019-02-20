#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import operator
from collections import defaultdict

with open ("all.txt", "r") as myfile:
    data=myfile.read()

sentences=data.lower().replace('\r',' ').replace('\n',' ').replace('?','.').replace('!','.').replace('“','.').replace('”','.').replace("\"",".").replace('‘',' ').replace('-',' ').replace('’',' ').replace('\'',' ').split(".")

def remove_empty_words(l):
    return list(filter(lambda a: a != '', l))

# key=list of words, as a string, delimited by space
#     (I would use list of strings here as key, but list in not hashable)
# val=dict, k: next word; v: occurrences
first={}
second={}
third={}

def update_occ(d, seq, w):
    if seq not in d:
        d[seq]=defaultdict(int)

    d[seq][w]=d[seq][w]+1

for s in sentences:
    words=s.replace(',',' ').split(" ")
    words=remove_empty_words(words)
    if len(words)==0:
        continue
    for i in range(len(words)):
        # only two words available:
        if i>=1:
            update_occ(first, words[i-1], words[i])
        # three words available:
        if i>=2:
            update_occ(second, words[i-2]+" "+words[i-1], words[i])
        # four words available:
        if i>=3:
            update_occ(third, words[i-3]+" "+words[i-2]+" "+words[i-1], words[i])

"""
print ("third table:")
for k in third:
    print (k)
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    s=sorted(third[k].items(), key=operator.itemgetter(1), reverse=True)
    print (s[:20])
    print ("")
"""

test="i can tell"
#test="who did this"
#test="she was a"
#test="he was a"
#test="i did not"
#test="all that she"
#test="have not been"
#test="who wanted to"
#test="he wanted to"
#test="wanted to do"
#test="it is just"
#test="you will find"
#test="you shall"
#test="proved to be"

test_words=test.split(" ")

test_len=len(test_words)
last_idx=test_len-1

def print_stat(t):
    total=float(sum(t.values()))

    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    s=sorted(t.items(), key=operator.itemgetter(1), reverse=True)
    # take 5 from each sorted table
    for pair in s[:5]:
        print ("%s %d%%" % (pair[0], (float(pair[1])/total)*100))

if test_len>=3:
    tmp=test_words[last_idx-2]+" "+test_words[last_idx-1]+" "+test_words[last_idx]
    if tmp in third:
        print ("* third order. for sequence:",tmp)
        print_stat(third[tmp])

if test_len>=2:
    tmp=test_words[last_idx-1]+" "+test_words[last_idx]
    if tmp in second:
        print ("* second order. for sequence:", tmp)
        print_stat(second[tmp])

if test_len>=1:
    tmp=test_words[last_idx]
    if tmp in first:
        print ("* first order. for word:", tmp)
        print_stat(first[tmp])
print ("")

