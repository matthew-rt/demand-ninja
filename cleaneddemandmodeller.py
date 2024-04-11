# this is a cleaned version of the demand model code, to consider repackaging into a simpler to use form
# %%
import demand_ninja
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


populationsitelocs = np.loadtxt("populationsitelocs.csv", delimiter=",", skiprows=1)
# this file has the population within each MERRA 2 grid square. It was created to line up with the
# sitelocs in the humid-temp-wind-comb folder, so the site loc numbers may not match up if you're using
# a different dataset.

# %%
desiredpopulation = (
    61389177  # this allows us to scale the population to the desired population
)
currentpopulation = np.sum(populationsitelocs[:, 3])
scalingfactor = currentpopulation / desiredpopulation
populationsitelocs[:, 3] = populationsitelocs[:, 3] / scalingfactor

# %%


# %%
# %%
folder = "C:/Users/SA0011/Documents/data"
datafolder = folder + "/humid-temp-wind-comb/"
solarfolder = folder + "/solar_comb/"
sitelocs = np.loadtxt(datafolder + "sitelocs.csv", delimiter=",", skiprows=1)


# %%

# now we've found the population at each site loc, we need to work out the BAIT and HDD for each location.

datadict = {}
for i in range(len(populationsitelocs)):
    if populationsitelocs[i, 3] == 0:
        continue
    code = int(populationsitelocs[i, 0])
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
    datadict[code] = codedata

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
    popsums = []
    for i in populationsitelocs:
        if i[3] == 0:
            continue
        popsums.append(i[3])
        code = int(i[0])
        data = datadict[code]
        selectedyear = data[startindex:endindex]
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
# %%
DHWdemand = (
    24 * 20 * 0.45
)  # elsewhere we've found that DHW demand is 45% of the base component. The 24 converts from GW to GWh
startdate = datetime.datetime(2000, 1, 1)
with open("demanddata.csv", "w") as file:
    # coverts date to string

    file.write(
        "Date,DHW Demand(GWh),Heating Demand(GWh),Heating COP, DHW COP, Temperature (C)\n"
    )
    for i in range(len(hdddemand)):
        datestring = startdate.strftime("%Y-%m-%d")
        file.write(
            f"{datestring},{DHWdemand},{hdddemand[i]*24},{heatingCOPs[i]},{dhwCOPs[i]},{temps[i]}\n"
        )
        startdate += datetime.timedelta(days=1)


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

heatingdemand = np.array(heatingdemand) + 20
plt.plot(gasdemand["Demand"], label="Actual Demand")
plt.plot(heatingdemand, label="Modelled Demand")
plt.legend()

plt.show()
# %%
