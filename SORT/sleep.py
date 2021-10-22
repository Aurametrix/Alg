import threading
from time import sleep

items = [5,2,1,3]

def sleep_sort(i):
  sleep(i)
  print(i)

threads = []
for i in items:
  thread = threading.Thread(target=sleep_sort, args=(i,))
  threads.append(thread)
  thread.start()

for t in threads:
  t.join()
