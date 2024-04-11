# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime

with open("demanddata.csv", "r") as file:
    data = file.read().strip().split("\n")
    headerline = data[0].split(",")
    splitdata = [i.split(",") for i in data[1:]]
    dates = [i[0] for i in splitdata]
    nondates = [i[1:] for i in splitdata]
    floatnondates = [[float(j) for j in i] for i in nondates]
    floatnondates = np.array(floatnondates)

# floatnondates[:, 2]
# floatnondates[:, 3] -= 0.85
# %%
DHWprofiles = np.loadtxt("hourlyDHW.csv", delimiter=",")

ASHPprofiles = np.loadtxt("hourlyASHP.csv", delimiter=",", skiprows=1)

# %%

heatbounds = [
    [-4.5, -1.5],
    [-1.5, 1.5],
    [1.5, 4.5],
    [4.5, 7.5],
    [7.5, 10.5],
    [10.5, 13.5],
    [13.5, 16.5],
    [16.5, 1000],
]
demandtimeseries = []
for day in range(len(floatnondates)):
    params = floatnondates[day]
    temp = params[4]
    for i, bound in enumerate(heatbounds):
        if temp >= bound[0] and temp < bound[1]:
            selectindex = i
            break

    selectedASHP = ASHPprofiles[:, selectindex]
    DHWdemand = DHWprofiles * params[0] / params[3]
    ASHPdemand = selectedASHP * params[1] / params[2]
    totaldemand = DHWdemand + ASHPdemand
    demandtimeseries.append(totaldemand)

# %%

demandtimeseries = np.hstack(demandtimeseries)


# %%
# works out seasonal coefficent of performance

totalelectricalpower = np.sum(demandtimeseries)
totalheatpower = np.sum(floatnondates[:, 0:2])
scop = totalheatpower / totalelectricalpower
print(scop)
# %%

# makes a datetime object from the first entry in dates
datastartdate = datetime.datetime.strptime(dates[0], "%Y-%m-%d")

samplestartdate = datetime.datetime(2010, 1, 1)
sampleenddate = datetime.datetime(2011, 1, 1)
startindex = (samplestartdate - datastartdate).days * 24
endindex = (sampleenddate - datastartdate).days * 24
selectedyeardemandseries = demandtimeseries[startindex:endindex]
print(np.max(selectedyeardemandseries))
print(np.min(selectedyeardemandseries))
print(np.sum(selectedyeardemandseries))
# %%
plt.plot(selectedyeardemandseries)
plt.xlabel("Hour")
plt.ylabel("Demand (GW)")
plt.title("2010 demand timeseries")
plt.show()


# %%
summedhotwaterdemand = np.sum(floatnondates[:, 0] / floatnondates[:, 3])
summedheatingdemand = np.sum(floatnondates[:, 1] / floatnondates[:, 2])
totaldemand = summedhotwaterdemand + summedheatingdemand
print(totaldemand - np.sum(demandtimeseries))
print(np.sum(demandtimeseries))
print(totaldemand)
# %%
