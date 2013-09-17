#simple recursive search algorithm in a maze/grid = list vs net represented by dictionary

"""
net = {'0':{'1':100, '2':300},
       '1':{'3':500, '4':500, '5':100},
       '2':{'4':100, '5':100},
       '3':{'5':20},
       '4':{'5':20},
       '5':{}
       }
"""

net = [[0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 2]]

# Algorithm walks the maze recursively by visiting each cell except already visited
# and avoiding walls

def search(x, y):
    if net[x][y] == 2:
        print 'found at %d,%d' % (x, y)
        return True
    elif net[x][y] == 1:
        print 'wall at %d,%d' % (x, y)
        return False
    elif net[x][y] == 3:
        print 'visited at %d,%d' % (x, y)
        return False
     
    print 'visiting %d,%d' % (x, y)
 
    # mark as visited
    net[x][y] = 3
 
    # explore neighbors clockwise starting by the one on the right
    if ((x < len(net)-1 and search(x+1, y))
        or (y > 0 and search(x, y-1))
        or (x > 0 and search(x-1, y))
        or (y < len(net)-1 and search(x, y+1))):
        return True
 
    return False
 
search(0, 0)
