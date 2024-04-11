# %%
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# %%
datafolder = "C:/Users/SA0011/Documents/data/generationdata/"

generatedata = pd.read_csv(datafolder + "2016--2022.csv")
# %%
# all generation data is equivalent to demand data, as generation will always equal demand
# we will be ignoring daylight savings time as it's annoying.
currentdatetime = datetime.datetime(2016, 1, 1)

totaldata = generatedata["TOTAL"]

monthdays = []

# %%
i = 0
hourcounter = 0
lastrealval = 27000
threshold = 15300
# we need to average the data to combine both half hourly values into one val
# we will do this by adding the two values together and dividing by 2
# the data is also noisy, and has some errors.
# We will replace any value which is too low with either the last value or the next value
# THhe threshold has been set based on the record for lowest demand, which is 15.3GW
datetimes = []
days = []
hourvals = []
monthdays = [[] for i in range(12)]
# %%
enddate = datetime.datetime(2020, 1, 1)
while i < len(totaldata):
    val1 = totaldata[i]
    val2 = totaldata[i + 1]
    if val1 < threshold:
        if val2 > threshold:
            val1 = val2
        else:
            val1 = lastrealval
    if val2 < threshold:
        if val1 > threshold:
            val2 = val1
        else:
            val2 = lastrealval

    aveval = (val1 + val2) / 2
    hourvals.append(aveval)
    i += 2
    hourcounter += 1
    if hourcounter == 24:
        hourcounter = 0
        days.append(hourvals)
        monthdays[currentdatetime.month - 1].append(hourvals)
        hourvals = []
        datetimes.append(currentdatetime)
        currentdatetime += datetime.timedelta(days=1)
        if currentdatetime >= enddate:  # stopping prior to 2020 due to COVID
            break
    lastrealval = aveval

# %%


jandays = monthdays[0]
flattendjandays = [val for day in jandays for val in day]
print(max(flattendjandays))
summedjandays = [np.sum(day) for day in jandays]

maxjanday = np.argmax(summedjandays)
day15 = jandays[15]
day15sum = np.sum(day15)
print(day15sum)
print(summedjandays[maxjanday])
warpfactor = summedjandays[maxjanday] / day15sum
warped15 = np.array(day15) * warpfactor
print(max(warped15))
plt.plot(day15, label="15th Jan")
plt.plot(warped15, label="15th Jan Warped")
plt.plot(jandays[maxjanday], label="Max Jan")
plt.legend()
plt.show()
jandaymaxes = [np.max(day) for day in jandays]
peakiestday = np.argmax(jandaymaxes)
# %%

janargsort = np.argsort(summedjandays)
# plots the 6 highest demand days in January
for i in range(1, 4):
    plt.plot(jandays[janargsort[-i]], label="Day " + str(i + 1))

plt.plot(jandays[peakiestday], label="Peakiest Day")
plt.legend()
plt.show()
# %%

plt.plot(monthdays[0][1])
plt.show()
# %%


for index, month in enumerate(monthdays):
    dailysums = [np.sum(day) for day in month]
    meansum = np.mean(dailysums)
    closesttomeansum = np.argmin(np.abs(dailysums - meansum))
    selectedday = month[closesttomeansum]
    selectedday = np.array(selectedday)
    selectedday = selectedday / np.sum(selectedday)
    with open(f"monthlyprofiles/{index+1}.csv", "w") as f:
        for val in selectedday:
            f.write(str(val) + "\n")
# %%
