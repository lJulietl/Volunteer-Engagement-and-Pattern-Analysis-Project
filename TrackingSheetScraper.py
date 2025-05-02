import pandas as pd

def days(tracking):
    day1 = tracking.iloc[3:60,1:4]
    day2 = tracking.iloc[3:60,5:8]
    day3 = tracking.iloc[3:60,9:12]
    day4 = tracking.iloc[3:60,13:16]
    day5 = tracking.iloc[3:60,17:20]
    day6 = tracking.iloc[3:61,21:24]
    day7 = tracking.iloc[3:61,25:28]
    dayList = [day1, day2, day3, day4, day5, day6, day7]
    return dayList


def weekday(day):
    stocking = day.iloc[3:11]
    hour10 = day.iloc[11:17]
    hour11 = day.iloc[17:23]
    hour12 = day.iloc[23:29]
    hour1 = day.iloc[29:35]
    hour2 = day.iloc[35:41]
    hour3 = day.iloc[41:47]
    closing = day.iloc[47:52]
    covers = day.iloc[52:57]
    shiftsList = [stocking, hour10, hour11, hour12, hour1, hour2, hour3, closing, covers]
    return shiftsList

def weekend(day):
    opening = day.iloc[10:14]
    closing = day.iloc[16:20]
    covers = day.iloc[52:57]
    shiftsList = [opening, closing, covers]
    return shiftsList


def entry(i):
    if not pd.isna(i.iloc[1]):
        shiftType = i.iloc[0]
        name = i.iloc[1]
        Attendance = i.iloc[2]
        entryList = [shiftType, name, Attendance]
        return entryList
    else:
        return None


def weekDF(file):
    tracking = pd.read_csv(file)
    col = ['Week', 'Day', 'Time', 'Name', 'Attendance', 'Shift Type']
    df = pd.DataFrame(columns = col)
    week = tracking.columns[1]
    dayList = days(tracking)
    for i in range(5):
        Xday = dayList[i]
        day = Xday.iloc[0,0]
        shiftsList = weekday(Xday)
        for j in range(9):
            if j == 0:
                time = 'Stocking Shift: 9:00 AM - 10:30 AM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        shiftType = [entryList[0], 'Stocking']
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], shiftType]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 1:
                time = '10:00 AM - 11:00 AM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 2:
                time = '11:00 AM - 12:00 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 3:
                time = '12:00 PM - 1:00 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 4:
                time = '1:00 PM - 2:00 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 5:
                time = '2:00 PM - 3:00 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 6:
                time = '3:00 PM - 4:00 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 7:
                time = 'Closing Shift: 4:00 PM - 4:45 PM'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        shiftType = [entryList[0], 'Closing']
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], shiftType]], columns = col)
                        df = pd.concat([df, newEntry])
            else:
                time = 'Shift Covers (variable)'
                shift = shiftsList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
    for i in range(5,7):
        Xday = dayList[i]
        day = Xday.iloc[0,0]
        shiftList = weekend(Xday)
        for j in range(3):
            if j == 0:
                time = 'Opening Shift: 12:00 PM - 1:00 PM'
                shift = shiftList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            elif j == 1:
                time = 'Closing Shift: 1:00 PM - 2:00 PM'
                shift = shiftList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
            else:
                time = 'Shift Covers (variable)'
                shift = shiftList[j]
                for idx, k in shift.iterrows():
                    entryList = entry(k)
                    if entryList is not None:
                        newEntry = pd.DataFrame([[week, day, time, entryList[1], entryList[2], entryList[0]]], columns = col)
                        df = pd.concat([df, newEntry])
    OutputFile = week.split(' ')[0] + '_' + week.split(' ')[1] + '.csv'                  
    df.to_csv(OutputFile, index=False)
    print('All done!')