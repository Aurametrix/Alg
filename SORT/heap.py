# MergeSort =  comparison-based, slower than quicksort but better in worst-case O(n log n)
# heap is a specialized tree-based data structure
def HeapSort(A):
     # create a heap out of given array of elements
     def heapify(A):
        start = (len(A) - 2) / 2
        while start >= 0:
            siftDown(A, start, len(A) - 1)
            start -= 1

     def siftDown(A, start, end):
        root = start
        while root * 2 + 1 <= end:
            child = root * 2 + 1
            if child + 1 <= end and A[child] < A[child + 1]:
                child += 1
            if child <= end and A[root] < A[child]:
                A[root], A[child] = A[child], A[root]
                root = child
            else:
                return

     heapify(A)
     end = len(A) - 1

     #print "pre-sort: ", A
     while end > 0:
        A[end], A[0] = A[0], A[end]
        siftDown(A, 0, end - 1)
        #print "inside sort: ", A
        end -= 1

import sys 
def main():
    l = [1, 6, 7, 2, 76, 45, 23, 4, 8, 12, 11]
    if (len(sys.argv) > 2):
        l = map(int,sys.argv[1:])        # whatever typed when running the program
    print "before sort", l
    HeapSort(l)
    print "after sort ", l
 
if __name__ == '__main__':
    main()