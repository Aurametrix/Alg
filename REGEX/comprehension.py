needle = raw_input("Enter a string: ")

# old-fashioned comprehension

if needle.endswith('ly') or needle.endswith('ed') or\
    needle.endswith('ing') or needle.endswith('ers'):
    print('It is valid')
else:
    print('It is invalid')

# more elegant comprehension

if any([needle.endswith(e) for e in ('ly', 'ed', 'ing', 'ers')]):
    print('It is valid')
else:
    print('It is invalid')