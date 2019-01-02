def any(S):
        for x in S:
            if x:
               return True
        return False

    def all(S):
        for x in S:
            if not x:
               return False
        return True
Combine these with generator expressions, and you can write things like these::

    any(x > 42 for x in S)     # True if any elements of S are > 42
    all(x != 0 for x in S)     # True if all elements if S are nonzero
