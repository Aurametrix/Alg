if __name__=="__main__":                      
    import sys
    from quicksort import quicksort
    lst = map(int,sys.argv[1:])        # Get typed numbers as list
    #print quicksort(lst)          # print out the sorted list as a list
    import string
    print string.join(map(str,quicksort(lst) ))    # Print as a string 
    