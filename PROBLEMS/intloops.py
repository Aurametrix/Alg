 from itertools import permutations
  
  # Initialization
  numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  primes = [3, 5, 7, 11, 13, 17]
  winners = []
  
  # Loop though the permutations, collecting only prime pairs
  for p in permutations(numbers):
    for i in range(1, 9):
      if p[i-1] + p[i] not in primes:
        break
    else:
      winners.append(p)
  
  # Print the results
  print(len(winners))
  for p in winners:
    print(p)
