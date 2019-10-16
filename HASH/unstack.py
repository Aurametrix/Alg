        tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
   ...:                      'foo', 'foo', 'qux', 'qux'],
   ...:                     ['one', 'two', 'one', 'two',
   ...:                      'one', 'two', 'one', 'two']]))
   ...: 

        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

        df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

        df2 = df[:4]
         
        df2
        
        stacked = df2.stack()

        stacked
        
        stacked.unstack()
        stacked.unstack(1)
        stacked.unstack(0)
        stacked.unstack('second')
        index = pd.MultiIndex.from_product([[2, 1], ['a', 'b']])
        df = pd.DataFrame(np.random.randn(4), index=index, columns=['A'])
        all(df.unstack().stack() == df.sort_index())
