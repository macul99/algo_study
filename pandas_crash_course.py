## df.drop_duplicates(subset=['a'])

## df.drop()
# columns
df.drop('a', axis=1) df.drop(columns='a') df.drop(columns=['a'])
# rows
df.drop(0) df.drop(index=0) df.drop(index=[0])

## df.dropna()
# columns
df.dropna(axis=1) df.dropna(how='all/any', axis=1) df.dropna(thresh=3, axis=1) # column with less than 3 valid values will be droped
# rows
df.dropna() df.dropna(subset=['a']) #subset for rows only 
df.drop(how='all/any') df.drop(thresh=3, inplace=True, ignore_index=True)

## df.fillna(
  value, # scalar/dict/Series/DataFrame, CANNOT be List
         # dict.key and Series.index matches column names of df
         # dataframe should match the shape of original df
)
df.fillna(0)
df.fillna({'a':0})
df.fillna(pd.Series([0,0], index=['a','b']))
df.fillna(pd.DataFrame([[1,2],[3,4]], columns=['a','b']))

## df.where(cond, other) # replace with 'other' value if not meeting 'cond', opposite of df.mask()
## df.mask(cond, other) # repalce with 'other' value if meeting 'cond'
df.where(df>0, -df) # cond is the same shape of original df, replace based on each element of the condition
df.where(df['a']>0, -df) # cond is a series, replace the whole row
