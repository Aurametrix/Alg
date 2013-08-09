# Bubblesort = repeated swapping, worst case O(n * n)
        
def bubble(list):
    length = len(list) - 1
    sorted = False  # the list is not sorted yet
    
    while not sorted:
        sorted = True  # Assuming the list is now sorted
        for i in range(length):
            if list[i] > list[i + 1]:
                sorted = False  # at least two elements are in the wrong order
                list[i], list[i+1] = list[i+1], list[i]  #swap
                