# %%
import numpy as np

with open("halfhourlyASHP.csv", "r") as file:
    data = file.read().strip().split("\n")
    headerline = data[0]
    splitdata = [i.split(",") for i in data[1:]]
    floatsplitdata = [[float(j) for j in i] for i in splitdata]

    floatsplitdata = np.array(floatsplitdata)
# %%
# floatsplitdata is 48 long, as its half hourly. We want to combine it into 24 hourly values, by summing the two half hourly values
hourlydata = []
for i in range(24):
    hourlydata.append(floatsplitdata[i * 2] + floatsplitdata[i * 2 + 1])

hourlydata = np.array(hourlydata)
# check that each column is still normalised (i.e. the sum of each column is 1)
print(np.sum(hourlydata, axis=0))
hourlydata = np.round(hourlydata, 5)
# %%

with open("hourlyASHP.csv", "w") as file:
    file.write(headerline + "\n")
    for i in hourlydata:
        file.write(",".join([str(j) for j in i]) + "\n")
# %%

with open("halfhourlyDHW.csv", "r") as file:
    data = file.read().strip().split("\n")
    splitdata = [i.split(",") for i in data]
    floatsplitdata = [[float(j) for j in i] for i in splitdata]

    floatsplitdata = np.array(floatsplitdata)

# %%

# floatsplitdata is 48 long, as its half hourly. We want to combine it into 24 hourly values, by summing the two half hourly values
hourlydata = []
for i in range(24):
    hourlydata.append(floatsplitdata[i * 2] + floatsplitdata[i * 2 + 1])

hourlydata = np.array(hourlydata)
hourlydata = hourlydata / np.sum(hourlydata, axis=0)
hourlydata = np.round(hourlydata, 5)
# %%
with open("hourlyDHW.csv", "w") as file:
    for i in hourlydata:
        file.write(",".join([str(j) for j in i]) + "\n")
# %%
