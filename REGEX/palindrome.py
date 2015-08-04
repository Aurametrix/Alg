def is_palindrome(s):
  return s == s[::-1]

def is_palindrome_r(s):
  if len(s) <= 1:
    return True
  elif s[0] != s[-1]:
    return False
  else:
    return is_palindrome_r(s[1:-1])
