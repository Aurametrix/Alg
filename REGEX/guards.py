def classify(pet):
  match pet:
    case Cat(age=age) if age % 2 == 0:
      print("This cat has an even age!")
    case Dog(name=name) if sorted(name) == list(name):
      print("This dog's name is in alphabetical order!")
    case _:
      print("I have nothing interesting to say about this pet.")

classify(Cat(4))
# prints "This cat has an even age!"

classify(Dog("abe"))
# prints "This dog's name is in alphabetical order!"

classify(Dog("fido"))
# prints "I have nothing interesting to say about this pet."
