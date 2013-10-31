
# In-place comparison sorting algorithm 
#  Best for computer memory as each element is moved no more than once
#  performance O(n^2)

def cyclesort( aList ):
    writes = 0
 
    for cs in range( len( aList ) - 1 ):
      # if the element in the list is out of place
      seeker = aList[cs]
      pos = cs
      # find the correct position (pos) of seeker
      for i in range( cs + 1, len( aList ) ):
        if aList[i] < seeker:
          pos += 1
 
      # if seeker is already in correct position, move on
      if pos == cs:
        continue
 
      # move index pos after duplicates if any
      while seeker == aList[pos]:
        pos += 1
 
      # switch in search of a right position.
 
      seeker = set_value( aList, seeker, pos )
      # track the number of writes
      writes += 1
 
      #  complete the current cycle  - until pos equal cs
 
      while pos != cs:
        # same as block of code above
        pos = cs
        for i in range( cs + 1, len( aList ) ):
          if aList[i] < seeker:
            pos += 1
 
        while seeker == aList[pos]:
          pos += 1
 
        seeker = set_value( aList, seeker, pos )
        writes += 1
 
#    return writes
    return aList 
 
def set_value( aList, data, ndx ):
    try:
      return aList[ndx]
    finally:
      aList[ndx] = data
