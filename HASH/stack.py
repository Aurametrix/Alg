# using lists as stack - last in first out
a = [66.25, 333, 333, 1, 1234.5]
print "original list:", a
print "333 appears:", a.count(333), " times while x appears ", a.count('x'), "times"
a.insert(2, -1)
# add an item to the end of the list, same as a[len(a):] = [x]
a.append(333)
print "affter inserting -1 at pos. 2 and appending 333", a
print a.index(333)  #index in the list of the first item with value 333
a.remove(333)
a.reverse()
print "after removing 333 and reversing", a
a.sort()
print a
a.pop()
print "after popping ", a
