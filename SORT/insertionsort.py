
def insertionsort(lst):
    for i in range(1, len(lst)):
        for j in range(i):
            if lst[i] < lst[j]:
                x = lst.pop(i)
                lst.insert(j, x)

    return lst


alist = [54,26,93,17,77,31,44,55,20]
insertionsort(alist)
print(alist)
