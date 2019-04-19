  def fun(name, **kwords):
        return 'name' in kwords       # always False

    def fun(name, /, **kwords):
        return 'name' in kwords       # True for fun(a, name=foo)

# An example using the str.format_map() builtin:

    def fun(fmt, **kwords):
        fmt.format_map(kwords)

    fun('format: {fmt}', fmt='binary')  # TypeError because fmt is reused
