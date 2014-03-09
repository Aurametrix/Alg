print "unpacking"
a, b, c = 1, 2, 3
print a, b, c
a, b, c = [1, 2, 3]
print a, b, c
a, b, c = (2 * i + 1 for i in range(3))
print a, b, c
a, (b, c), d = [1, (2, 3), 4]
print a, b,c,d

print "\nunpacking for swapping"
a, b = 1, 2
a, b = b, a
print a, b

# 
# print "extended unpacking - intriduced in python 3/n"
# a, *b, c = [1, 2, 3, 4, 5]
# print a,b, c

print "\nnegative indexing for a list from 0 to 10"
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print "minus first index:", a[-1], "; minus third index:", a[-3]

print "list slices:"
print "slices 2 to 8:", a[2:8]

print "slices from -4 to -2:", a[-4:-2]

print "a[::2] ", a[::2]
print "a[::3] ", a[::3]
print "a[::-1] ", a[::-1]

a[2:3] = [0, 0]
print "replacing a slice 2-3 with 0,0: ", a

a[1:1] = [8, 9]
print "replacing a slice 1-1 with 8,9: ", a

a[1:-1] = []
print "replacing a slice 1:-1 with empty list: ", a

print "\n=Naming slices"
a = [0, 1, 2, 3, 4, 5]
LASTTHREE = slice(-3, None)
print LASTTHREE
slice(-3, None, None)
print a[LASTTHREE]
