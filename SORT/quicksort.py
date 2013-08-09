# QuickSort = partition-exchange sort O(n log n)
def quicksort(list):   
    if list == []: 
        return []
    else:
        pivot = list[0]  #pick an element
# create empty lists 'less' and 'greater'
#partition: reorder to put all lessers before, 
#all with greater values after 
        lesser = quicksort([x for x in list[1:] if x < pivot])
        greater = quicksort([x for x in list[1:] if x >= pivot])
        return lesser + [pivot] + greater