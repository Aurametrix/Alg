import d6tjoin.top1
import d6tjoin.utils

#************************
# pre join diagnostics
#************************

# check join quality => none of the ids match

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin()

  key left key right  all matched  inner  left  right  outer  unmatched total  unmatched left  unmatched right
0       id        id        False      0    10     10     20               20              10               10
1     date      date         True    366   366    366    366                0               0                0
2  __all__   __all__        False      0  3660   3660   7320             7320            3660             3660

#************************
# best match join on id
#************************

result = d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'],
    exact_left_on=['date'],exact_right_on=['date']).merge()

result['merged'].head(2)

        date        id   val1 id_right  val1_right   val2
0 2010-01-01  e3e70682  0.020   3e7068       0.020  0.034
1 2010-01-01  f728b4fa  0.806   728b4f       0.806  0.849

#************************
# debug best matches
#************************

result['top1']['id'].head(2)

         date __top1left__ __top1right__  __top1diff__ __matchtype__
10 2010-01-01     e3e70682        3e7068             2     top1 left
34 2010-01-01     e443df78        443df7             2     top1 left

#************************
# customize similarity fct
#************************
import affinegap

result = d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'], 
    fun_diff=[affinegap.affineGapDistance]).merge()

#************************
# token-based substring clusters and joins
#************************
dftoken=d6tjoin.utils.splitcharTokenCount(df2['id'])
