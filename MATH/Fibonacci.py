def hybrid_fib(n):
    if n <= len(small_fib):
        return small_fib[n]
    elif n <= 2**12:
        return cython_fib(n)
    else:
        return gmp_fib(n)
