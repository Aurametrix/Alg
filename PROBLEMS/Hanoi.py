def hanoi(ndisks, startPeg=1, endPeg=3):
    if ndisks:
        hanoi(ndisks-1, startPeg, 6-startPeg-endPeg)
        print "Move disk %d from peg %d to peg %d" % (ndisks, startPeg, endPeg)
        hanoi(ndisks-1, 6-startPeg-endPeg, endPeg)
 
hanoi(ndisks=4)

def hanoi(height, left='left', right='right', middle='middle'):
    if height:
        hanoi(height - 1, left, middle, right)
        print(left, '=>', right)
        hanoi(height - 1, middle, right, left)
