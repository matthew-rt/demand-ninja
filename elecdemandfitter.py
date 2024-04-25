# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

demanddata = np.load("demanddata.npy")
Hdds = np.load("hdds.npy")


Hdds = Hdds[-len(demanddata) :]
Hdds[Hdds > 13] = 13
weekenddirac = np.zeros(len(demanddata))
sundayindices = np.arange(0, len(demanddata), 7)
mondayindices = np.arange(1, len(demanddata), 7)
tuesdayindices = np.arange(2, len(demanddata), 7)
wednesdayindices = np.arange(3, len(demanddata), 7)
thursdayindices = np.arange(4, len(demanddata), 7)
fridayindices = np.arange(5, len(demanddata), 7)
saturdayindices = np.arange(6, len(demanddata), 7)

weekenddirac[saturdayindices] = 1
weekenddirac[sundayindices] = 1
# %%
dayarray = np.array([i for i in range(0, len(demanddata))])
sinusoidal = np.sin(dayarray / 365 * np.pi)
sinusoidal = sinusoidal**4
plt.plot(sinusoidal)
plt.show()
sinusoidal[sinusoidal < 0] *= -1

weekenddemand = []
weekdaydemand = []
index = 1
while index < len(demanddata) - 7:
    weekdaydemand.append(np.mean(demanddata[index : index + 5]))
    weekenddemand.append(np.mean(demanddata[index + 5 : index + 7]))
    index += 7

weekenddemand = np.array(weekenddemand)
weekdaydemand = np.array(weekdaydemand)
differences = weekenddemand - weekdaydemand
weeknumber = [i for i in range(0, len(differences))]
plt.plot(weeknumber, differences)
plt.show()
# plt.plot(sinusoidal)
# %%
# def func(x, Pbase, Pheat):
#     return Pbase + Pheat * x


# %%
def func(x, Pbase, Pheat, weekendfactor):
    return Pbase + Pheat * x + weekendfactor * (1 + sinusoidal) * weekenddirac


popt, pcov = curve_fit(func, Hdds, demanddata)

predictedvals = func(Hdds, *popt)
# find correlation coefficient
r = np.corrcoef(predictedvals, demanddata)[0, 1]
print("Correlation coefficient: ", r)
# print(popt)
plt.plot(demanddata, label="Actual demand")
plt.plot(func(Hdds, *popt), label="Predicted demand")

plt.xlabel("Days")
plt.ylabel("Demand (MW)")
# now we want to plot the days of the week on a scatter overlaying the data

# the first day of 2017 was a sunday


# plt.scatter(sundayindices, sundaydata, label="Sunday")
# plt.scatter(mondayindices, mondaydata, label="Monday")
# plt.scatter(tuesdayindices, tuesdaydata, label="Tuesday")
# plt.scatter(wednesdayindices, wednesdaydata, label="Wednesday")
# plt.scatter(thursdayindices, thursdaydata, label="Thursday")
# plt.scatter(fridayindices, fridaydata, label="Friday")
# plt.scatter(saturdayindices, saturdaydata, label="Saturday")

plt.legend()
plt.title("Predicted vs Actual Daily Demand 2017-2019")
plt.show()
