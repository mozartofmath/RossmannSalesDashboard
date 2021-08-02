import bisect

def weekends(x):
    if x >= 6:
        return 1
    return 0

def time_of_month(x):
    if x <= 10:
        return 0
    if x <= 20:
        return 1
    return 2

def label_holidays(x):
    if x in [0,'0','a','b','c']:
        return [0,'0','a','b','c'].index(x)
    return 5

def days_from_holiday(dates, holidays):
    days_till, days_after = [], []
    for day in dates:
        ind = bisect.bisect(holidays, day)
        #print(ind)
        if ind == 0:
            days_till.append((holidays[ind] - day).days)
            days_after.append(14)
        elif ind == len(holidays):
            days_till.append(14)
            days_after.append((day - holidays[ind - 1]).days)
        else:
            days_till.append((day - holidays[ind - 1]).days)
            days_after.append((holidays[ind] - day).days)
    return days_till, days_after