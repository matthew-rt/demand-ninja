# %%
import demand_ninja
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# %%
folder = "C:/Users/SA0011/Documents/data"
datafolder = folder + "/humid-temp-wind-comb/"
solarfolder = folder + "/solar_comb/"
sitelocs = np.loadtxt(datafolder + "sitelocs.csv", delimiter=",", skiprows=1)

districts = pd.read_csv(
    "districts2.csv"
)  # districts is a list of all local authority districts and their coords

# renames the official code column to just code
districts.rename(
    columns={"Official Code Local authority district": "Code"}, inplace=True
)
popestimates = pd.read_csv(
    "popestimates.csv"
)  # popestimates is a list of all local authority districts and their populations
# %%
popcodedict = {}
popcodes = popestimates[
    "Code"
].tolist()  # popcodes is a list of all the local authority district codes
popallages = popestimates[
    "All ages"
].tolist()  # popallages is a list of all the populations

popallages = [
    int(i.replace(",", "")) for i in popallages
]  # removes commas from the population numbers and converts them to integers

for i in range(len(popcodes)):
    popcodedict[popcodes[i]] = popallages[
        i
    ]  # creates a dictionary with the district codes as keys and the populations as values


districts["closest site"] = np.zeros(len(districts))
districts["population"] = np.zeros(len(districts))
# this iterates through the districts, and then compares them to site locs, our list of all our generation data points
# it then finds the closest site to each district and assigns it to the closest site column
for i in range(len(districts)):
    lat, long = districts["Geo Point"][i].split(",")
    lat = float(lat)
    long = float(long)
    distancetosites = np.linalg.norm(sitelocs[:, 1:3] - np.array([lat, long]), axis=1)
    districts["closest site"][i] = np.argmin(distancetosites)
    districts["population"][i] = popcodedict[districts["Code"][i]]


# creates a list of the site locs and the summed population of the districts at that site loc

sitecodes = districts["closest site"].tolist()

codes = np.unique(sitecodes)
# this gets us a list of all the site codes that correspond to a population centre

# now we want to work out the population at each site loc
popsums = np.zeros(len(codes))
for i in range(len(codes)):
    popsums[i] = np.sum(districts[districts["closest site"] == codes[i]]["population"])
# %%

# now we've found the population at each site loc, we need to work out the BAIT and HDD for each location.


# first we'll load in all the data for each location. It can then be subsampled for the desired years

codedatalist = []
for code in codes:

    humidtempwind = np.loadtxt(
        datafolder + str(int(code)) + ".csv",
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
    )[366 * 24 :]
    # the wind data starts in 1980, but the solar data starts in 1981, so we need to remove the first year of solar data
    solar = np.loadtxt(
        solarfolder + str(int(code)) + ".csv", delimiter=",", skiprows=1, usecols=(1)
    )[:-24]
    # the solar data goes one day too far, so we need to remove the last day of solar data
    solar = solar.reshape(len(solar), 1)
    # reshapes solar into a column vector
    codedata = np.hstack((humidtempwind, solar))
    # combines the data
    codedatalist.append(codedata)


# the data is now loaded in and combined.
# %%
years = [2000 + i for i in range(20)]
yearlydemandtimeseries = []
variablecomps = []
datastart = datetime.datetime(
    1981, 1, 1, 0, 0, 0
)  # this may need to be changed to suit your data
DHWtemp = 50
yearlydhwCOPs = []
yearlyheatingCOPs = []
yearlytemps = []
for year in years:
    starttime = datetime.datetime(year, 1, 1, 0, 0, 0)
    startindex = (starttime - datastart).days * 24
    endtime = datetime.datetime(year + 1, 1, 1, 0, 0, 0)
    endindex = (endtime - datastart).days * 24
    hdds = []
    dhwCOPs = []
    heatingCOPS = []
    temps = []
    for indexnum in range(len(codes)):
        data = codedatalist[indexnum]
        selectedyear = data[startindex:endindex]
        # the data is hourly. We want to average it to daily
        siteyeardata = []
        for i in range(len(selectedyear) // 24):
            daydata = selectedyear[i * 24 : (i + 1) * 24]
            averageddata = np.mean(daydata, axis=0)
            siteyeardata.append(averageddata)
        siteyeardata = np.array(siteyeardata)
        dataframe = pd.DataFrame(
            siteyeardata,
            columns=[
                "temperature",
                "wind_speed_2m",
                "humidity",
                "radiation_global_horizontal",
            ],
        )
        dataframe["temperature"] = dataframe["temperature"] - 273.15
        dataframe["humidity"] = dataframe["humidity"] * 1000
        baits = demand_ninja.core._bait(dataframe, 0.5, 0.01, -0.18, 0.051)
        baits = baits.to_numpy()
        hdd = 13.3 - baits
        hdd[hdd < 0] = 0
        hdds.append(hdd)
        heatsinktemp = 40 - 1 * (dataframe["temperature"])
        heatingtempdelta = heatsinktemp - (dataframe["temperature"])
        DHWtempdelta = DHWtemp - (dataframe["temperature"])
        heatingtempdelta[heatingtempdelta < 15] = 15
        DHWtempdelta[DHWtempdelta < 15] = 15
        heatingCOP = 6.08 - 0.09 * heatingtempdelta + 0.0005 * heatingtempdelta**2
        dhwCOP = 3.5 - 0.05 * DHWtempdelta + 0.0005 * DHWtempdelta**2
        heatingCOPS.append(heatingCOP)
        dhwCOPs.append(dhwCOP)
        temps.append(dataframe["temperature"])

    # we now need to do a weighted average of the HDDs, heatingCOPs, dhwCOPs and temps
    averagedhdds = np.average(hdds, axis=0, weights=popsums)
    averagedheatingCOPs = np.average(heatingCOPS, axis=0, weights=popsums)
    averageddhwCOPs = np.average(dhwCOPs, axis=0, weights=popsums)
    averagedtemps = np.average(temps, axis=0, weights=popsums)

    heatingpower = 140
    pbase = 20
    heatingdemand = [np.sum(popsums) * i * heatingpower / (10**9) for i in averagedhdds]

    summedvariablecomps = np.sum(
        [np.sum(popsums) * i * heatingpower / (10**9) for i in averagedhdds]
    )
    variablecomps.append(summedvariablecomps)
    yearlydemandtimeseries.append(heatingdemand)
    yearlydhwCOPs.append(averageddhwCOPs)
    yearlyheatingCOPs.append(averagedheatingCOPs)
    yearlytemps.append(averagedtemps)


# %%
# flattens the demand, heatingCOP, dhwCOP and temps
hdddemand = [item for sublist in yearlydemandtimeseries for item in sublist]
dhwCOPs = [item for sublist in yearlydhwCOPs for item in sublist]
heatingCOPs = [item for sublist in yearlyheatingCOPs for item in sublist]
temps = [item for sublist in yearlytemps for item in sublist]
#%%
DHWdemand=24*20*0.45 #elsewhere we've found that DHW demand is 45% of the base component. The 24 converts from GW to GWh
startdate=datetime.datetime(2000,1,1)
with open("demanddata.csv", "w") as file:
    #coverts date to string

    file.write("Date,DHW Demand(GWh),Heating Demand(GWh),Heating COP, DHW COP, Temperature (C)\n")
    for i in range(len(hdddemand)):
        datestring=startdate.strftime("%Y-%m-%d")
        file.write(f"{datestring},{DHWdemand},{hdddemand[i]*24},{heatingCOPs[i]},{dhwCOPs[i]},{temps[i]}\n")
        startdate+=datetime.timedelta(days=1)

    
# %%
basecomponent = 20 * 365.25
meanvariablecomponent = np.mean(variablecomps)
dhwheatingcomp = 0.3
stddevvariablecomponent = np.std(variablecomps)

basecomponentDHWfrac = meanvariablecomponent * dhwheatingcomp / basecomponent
print(basecomponentDHWfrac)
stddevvariablecomponentDHWfrac = (
    stddevvariablecomponent * dhwheatingcomp / basecomponent
)

# %%
hdddates = [
    datetime.datetime(2019, 1, 1) + datetime.timedelta(days=i) for i in range(365)
]

gasdemand = pd.read_csv("2019demand.csv")
# converts "Applicable For" column to date
gasdemand["Date"] = pd.to_datetime(gasdemand["Date"], format="%Y-%m-%d")
# sorts dataframe by date


plt.plot(gasdemand["Demand"], label="Actual Demand")
plt.plot(hdddemand)
plt.show()
# %%
