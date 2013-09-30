# using lists as queue - first in first out
# performance using list implementation will be poor because list objects are optimized
# for fast fixed-length operations and incur O(n) memory movement costs
# While appends and pops from the end of list are fast, 
# inserts insert(0, v) or pops pop(0) from the beginning of a list are slow 
# as all of the other elements have to be shifted by one

a = [66.25, 333, 333, 1, 1234.5]
print "original list:", a
a.pop(0)
print "after popping the first element", a
a.insert(0, -1)
print "after inserting ", a

# using a deque is much faster.
from collections import deque
queue = deque([66.25, 333, 333, 1, 1234.5])
queue.append(-1)           # add -1
print "after inserting ", queue
queue.append("First")          # First arrives
print "after inserting ", queue
queue.popleft()                 # The first to arrive now leaves
print "after popleft ", queue
queue.popleft()                 # The second to arrive now leaves
print "after popleft ", queue