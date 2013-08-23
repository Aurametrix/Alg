# MergeSort =  comparison-based, divide-and-conquer O(n log n)
 
def mergeSort(listtosort):
    if len(listtosort) <= 1:
        return listtosort
 
    mIndex = len(listtosort) / 2
    left = mergeSort(listtosort[:mIndex])
    right = mergeSort(listtosort[mIndex:])
 
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] > right[0]:   
            result.append(right.pop(0))
        else:
            result.append(left.pop(0))
 
    if len(left) > 0:
        result.extend(mergeSort(left))
    else:
        result.extend(mergeSort(right))
 
    return result
 
import sys 
def main():
    l = [1, 6, 7, 2, 76, 45, 23, 4, 8, 12, 11]
    if (len(sys.argv) > 2):
        l = map(int,sys.argv[1:])        # Get typed numbers as list
    print "before sort", l
    sortedList = mergeSort(l)
    print "after sort ", sortedList
 
if __name__ == '__main__':
    main()