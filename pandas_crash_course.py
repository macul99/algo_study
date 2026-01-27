### Based on Pandas 2.3

## df.convert_dtypes() ds.convert_dtypes() # convert input data to proper dtypes by correctly handling pd.NA values
# very useful for reading dataset from IO. Otherwise, the datatype may be 'object'.

## df.drop_duplicates(subset=['a'])

## df.drop()
# columns
df.drop('a', axis=1) df.drop(columns='a') df.drop(columns=['a'])
# rows
df.drop(0) df.drop(index=0) df.drop(index=[0])

## Calculations with missing data
# When summing data, pd.NA values or empty data will be treated as zero
# When taking the product, pd.NA values or empty data will be treated as 1
# Cumulative methods like cumsum() and cumprod() ignore pd.NA values by default preserve them in the result. This behavior can be changed with 'skipna'

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
df.fillna(df.mean())

## df.ffill(limit) df.bfill(limit)
df.ffill()
df.bfill(limit=3)

## df.interpolate(method) ds.interpolate(method) # fillna by interpolation, methods = ["linear", "quadratic", "cubic", "barycentric", "phcip", "akima"]
df.interpolate(method='spine', order=2)
df.interpolate(method='polynomial', order=2)

## df.where(cond, other) # replace with 'other' value if not meeting 'cond', opposite of df.mask()
## df.mask(cond, other) # repalce with 'other' value if meeting 'cond'
df.where(df>0, -df) # cond is the same shape of original df, replace based on each element of the condition
df.where(df['a']>0, -df) # cond is a series, replace the whole row

# df.sort_values(['a'], ascending, key, ignore_index)


### Series
# Series of String
ds.str.len() # return a series of the string len for each item, slower than df.apply(lambda x: len(x))
ds.str.match(pattern)] # match regex pattern




