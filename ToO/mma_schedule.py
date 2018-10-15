#!/usr/bin/python
""" mma_schedule module
"""

from datetime import datetime


def mma_on_duty(filename_txt):

    gwac_mma = "None"
    timenow = str(datetime.utcnow())
    year = str(timenow.split()[0]).split("-")[0]
    month = str(timenow.split()[0]).split("-")[1]
    day = str(timenow.split()[0]).split("-")[2]
    hours = str(timenow.split()[1]).split(":")[0]
    minutes = str(timenow.split()[1]).split(":")[1]

    for line in open(filename_txt):
        #FIRST
        line_split = line.split()
        year_duty_start = str(line_split[1])
        month_duty_start = str(line_split[2])
        day_duty_start = str(line_split[3])
        mma_duty_morning = str(line_split[-2])

        #SECOND
        year_duty_stop = str(line_split[13])
        month_duty_stop = str(line_split[14])
        day_duty_stop = str(line_split[15])
        hour_duty_stop = str(line_split[16].split(":")[0])
        minute_duty_stop = str(line_split[16].split(":")[1])
        mma_duty_afternoon = str(line_split[-1])

        if (year >= year_duty_start) and (year <= year_duty_stop):
            if (month >= month_duty_start) and (month <= month_duty_stop):
                if (day >= day_duty_start) and (day <= day_duty_stop):
                    if (float(hours)*60.0+float(minutes)*1.0) <= (float(hour_duty_stop)*60.0+\
                         float(minute_duty_stop)):
                        gwac_mma = mma_duty_morning
                    else:
                        gwac_mma = mma_duty_afternoon

    return gwac_mma


def GWAC_onduty(filenametxt):
    gwac_duty = "None"
    timenow = str(datetime.utcnow())
    year = str(timenow.split()[0]).split("-")[0]
    month = str(timenow.split()[0]).split("-")[1]
    day = str(timenow.split()[0]).split("-")[2]

    for line in open(filenametxt):
        #FIRST
        line_split = line.split()
        year_duty_start = str(line_split[1])
        month_duty_start = str(line_split[2])
        day_duty_start = str(line_split[3])
        year_duty_stop = str(line_split[5])
        month_duty_stop = str(line_split[6])
        Day_duty_stop = str(line_split[7])
        mma_gwac = str(line_split[8])

        if (year >= year_duty_start) and (year <= year_duty_stop):
            if (month >= month_duty_start) and (month <= month_duty_stop):
                if (float(month_duty_stop)-float(month_duty_start)) > 0:
                    if (float(day) >= float(day_duty_start)) and ((float(day)) <= (float(Day_duty_stop\
                       )+float(day_duty_start))):
                        gwac_duty = mma_gwac
                else:
                    if (float(day) >= float(day_duty_start)) and (float(day) <= float(Day_duty_stop)):
                        gwac_duty = mma_gwac
    return gwac_duty
