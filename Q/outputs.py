list = ['a', 'b', 'c', 'd', 'e']
print list[4:]
print list[10:]

# python 2 vs python 3
def div1(x,y):
    print "%s/%s = %s" % (x, y, x/y)

def div2(x,y):
    print "%s//%s = %s" % (x, y, x//y)

div1(5,2)
div1(5.,2)
div2(5,2)
div2(5.,2.)

# By default, Python 2 automatically performs integer arithmetic if both operands are int
# 5/2 yields 2, while 5./2 yields 2.5.  // performs integer division
# to override: from __future__ import division 
# Python3 doesn't perform integer arithmetic

def extendList(val, list=None):
# w/t None & if list is assumed to be 'a'
    if list is None:
        list = []
    list.append(val)
    return list

list1 = extendList(10)
list2 = extendList(123,[])
list3 = extendList('a')

print "list1 = %s" % list1
# print "list1 = ", list1

print "list2 = %s" % list2
print "list3 = %s" % list3
