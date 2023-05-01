#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:42:28 2023

@author: matthewmcmurry
"""

import pandas as pd
import datetime
from datetime import datetime as dt
import numpy as np

path = '/path/to/folder'
df = pd.read_csv('/path/to/file.csv')

import datetime
import calendar

def generate_timestamps(year, day, month, num_intervals):
    timestamps = []
    dt = datetime.datetime(year, month, day)
    timestamps.append(calendar.timegm(dt.utctimetuple()))
    for i in range(1, num_intervals):
        dt += datetime.timedelta(weeks=1)
        timestamps.append(calendar.timegm(dt.utctimetuple()))
    return timestamps
# Ex: generate 3 years of weekly timestamps starting on 1/1/20
timestamps_list = generate_timestamps(2020, 1, 1, 156)


def generate_segments(timestamp_list, path, dates):
    #set path to csv file minus '.csv'
    dates = dates
    df = pd.read_csv(path + '.csv')
    i = 1
    for item in timestamp_list:
        while i < len(timestamp_list):
            start = timestamp_list[i-1]
            end = timestamp_list[i]
            df_filtered = df[(df['created_utc'] >= start) & (df['created_utc'] < end)].reset_index(drop=True)
            df_filtered.to_csv(path + '_Week_' + str(i) + '.csv', line_terminator='\r\n', index=False)
            print('Exported Segment ' + str(i))
            i+=1
    return "Export complete"




