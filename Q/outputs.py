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

# class variables are internally handled as dictionaries
# If a variable name is not found in the dictionary of the current class, its hierarchy 
# such as parent classes - are searched until the referenced variable name is found 
# if the referenced variable name is not found in the class itself or anywhere in 
# its hierarchy, an AttributeError occurs

class Parent(object):
    x = 1

# setting x = 1 in the Parent class makes the class variable x (with a value of 1) 
# referenceable in that class and any of its children. 

class Child1(Parent):
    pass

class Child2(Parent):
    pass

print Parent.x, Child1.x, Child2.x
Child1.x = 2
print Parent.x, Child1.x, Child2.x
Parent.x = 3
print Parent.x, Child1.x, Child2.x