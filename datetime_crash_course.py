import numpy as np
import pandas as pd # version 2
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo # python 3.9 and above
import sys

assert pd.__version__.startswith('2')
assert sys.version_info >= (3, 9)  # zoneinfo added in Python 3.9

# Don't use datetime.utcnow(), it is deprecated and also causing confusion
# both datetime.now() and datetime.now(timezone.utc) are preferred
print(datetime.now(), datetime.now(timezone.utc))
print(datetime.now().hour, datetime.now(timezone.utc).hour)
############# !!!!!!!!! take note that both will return the same timestamp !!!!!!!!! ############
print(datetime.now().timestamp(), datetime.now(timezone.utc).timestamp())
############# Use ZoneInfo to Specify other timezone ############
print(datetime.now(ZoneInfo('Asia/Singapore')).timestamp(), datetime.now(ZoneInfo('UTC')).timestamp())
# Use astimezone() to convert timezone, timestamp remain the same
print(datetime.now().astimezone(ZoneInfo("UTC")).timestamp()==datetime.now().astimezone(ZoneInfo("Asia/Singapore")).timestamp())
# Use replace() to set new timezone without convertion, timestamp will change
print(datetime.now().replace(tzinfo=ZoneInfo("UTC")).timestamp()==datetime.now().replace(tzinfo=ZoneInfo("Asia/Singapore")).timestamp())

print(datetime.now().timestamp()==datetime.now(timezone.utc).timestamp())
print(datetime.now().hour==datetime.now(timezone.utc).hour)
print(datetime.now(ZoneInfo('Asia/Singapore')).timestamp()==datetime.now(ZoneInfo('UTC')).timestamp())

# use pd.to_datetime to convert datetime.datetime to pandas.Timestamp
print(pd.to_datetime(datetime.now()), pd.to_datetime(datetime.now(timezone.utc)))
print(pd.to_datetime(datetime.now()).hour, pd.to_datetime(datetime.now(timezone.utc)).hour)
#### !!!! Take Note the converted timestamp is NOT SAME anymore !!!!! ##########
print(pd.to_datetime(datetime.now()).timestamp(), pd.to_datetime(datetime.now(timezone.utc)).timestamp())
#### !!!! Take Note the Only TZ Localized with CORRECT TZ, then timestamp is SAME !!!!! ##########
print(pd.to_datetime(datetime.now()).tz_localize('Asia/Singapore').timestamp(), pd.to_datetime(datetime.now(timezone.utc)).timestamp())
#### !!!! Localized with WRONG TZ, then timestamp is not the SAME !!!!! ##########
print(pd.to_datetime(datetime.now()).tz_localize('Asia/Jakarta').timestamp(), pd.to_datetime(datetime.now(timezone.utc)).timestamp())

################### CONCLUSION ########################
# datetime always tz aware even simply call datetime.now()
# date/hour/minute/second refers to local timestamp and timestamp always refers to UTC
# *** always return UTC timestamp regardless the timezone specified
# pandas.timestamp behaviour depends on whether it is tz-aware
# if it is tz-aware returned timestamp will refer to UTC
# if not tz-aware, timestamp is converted from date/hour/minute/second directly
# and SHOULD NOT COMPARE this timestamp with tz-aware timestamp
# Note: pd.to_datetime(datetime.now()) is NOT tz-aware

## pd.tz_localize() === datetime.replace()
# use date/hour/minute/second and replaced tz to create new datetime, ignore original tz and timestamp value
print(pd.to_datetime(datetime.now()).tz_localize("UTC").timestamp(), 
      datetime.now().replace(tzinfo=ZoneInfo("UTC")).timestamp())
print(pd.to_datetime(datetime.now()).tz_localize("Asia/Singapore").timestamp(), 
      datetime.now().replace(tzinfo=ZoneInfo("Asia/Singapore")).timestamp())
## pd.tz_convert() === datetime.astimezone()
print(pd.to_datetime(datetime.now()).tz_localize("UTC").tz_convert("Asia/Singapore").timestamp(), 
      datetime.now().replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Singapore")).timestamp())
## take note difference of below code, datetime.now() aware of system timezone which is Asia/Singapore
print(pd.to_datetime(datetime.now()).tz_localize("UTC").tz_convert("Asia/Singapore").timestamp(), 
      datetime.now().astimezone(ZoneInfo("Asia/Singapore")).timestamp())
print(pd.to_datetime(datetime.now()).tz_localize("Asia/Singapore").timestamp(), 
      datetime.now().astimezone(ZoneInfo("Asia/Singapore")).timestamp() # timestamp will not change when using astimezone
      )

print(pd.to_datetime(datetime.now())==pd.to_datetime(datetime.now(timezone.utc)))
print(pd.to_datetime(datetime.now()).hour==pd.to_datetime(datetime.now(timezone.utc)).hour)
#### !!!! Take Note the converted timestamp is NOT SAME anymore !!!!! ##########
print(pd.to_datetime(datetime.now()).timestamp()==pd.to_datetime(datetime.now(timezone.utc)).timestamp())
#### !!!! Take Note the Only TZ Localized with CORRECT TZ, then timestamp is SAME !!!!! ##########
print(pd.to_datetime(datetime.now()).tz_localize('Asia/Singapore').timestamp()==pd.to_datetime(datetime.now(timezone.utc)).timestamp())
#### !!!! Localized with WRONG TZ, then timestamp is not the SAME !!!!! ##########
print(pd.to_datetime(datetime.now()).tz_localize('Asia/Jakarta').timestamp()==pd.to_datetime(datetime.now(timezone.utc)).timestamp())


# use to_pydatetime() to convert pd.timestamp to datetime.datetime
# However, the usage is depends on whether its data type
# Series use dt, Note: after to_pydatetime() it will become numpy array and cannot use .dt anymore
# datetimeIndex use .to_pydatetime() directly, Note: after to_pydatetime() it will become numpy array and cannot use .dt anymore
# single timestamp use .to_pydatetime() as well

df_idx = pd.to_datetime(['2024-06-15 08:00', '2024-06-16 09:30'])
print(type(df_idx))
print(type(df_idx.tz_localize('UTC')))
print(type(df_idx.tz_localize('UTC').tz_convert('Asia/Singapore')))
print(type(df_idx.to_pydatetime()))
print(type(df_idx.tz_localize('UTC').to_pydatetime()))

ds = pd.Series(df_idx)
print(type(ds))
print(type(ds.dt.tz_localize('UTC')))
print(type(ds.dt.tz_localize('UTC').dt.tz_convert('Asia/Singapore')))
print(type(ds.dt.to_pydatetime()))
print(type(ds.dt.tz_localize('UTC').dt.to_pydatetime()))


# below code will fail since to_pydatatime() will output numpy array
ds.dt.tz_localize('UTC').dt.to_pydatetime().dt.tz_convert("Asia/Singapore")
