def stalin_sort(arr):
  if arr == []
    return arr
  
  max_val = arr[0]
  sorted_arr = []

  for val in arr:
    if val >= max_val:
      sorted_arr.append(val)
      max_val = val
    else:
      print(f"{val} sent to Gulag")
  
return sorted_arr
