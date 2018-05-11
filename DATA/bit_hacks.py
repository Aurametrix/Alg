def int_to_bin(num, bits=8):
 r = ''
 while bits:
  r = ('1' if num&1 else '0') + r
  bits = bits - 1
  num = num >> 1
 print r
