# %%
import numpy as np
import demand_ninja
import pyproj as proj
import datetime
import pandas as pd
import matplotlib.pyplot as plt


class demand:
    """This class exists as a wrapper to generate demand time series, and the other data required to generate it.
    This function requires that the demand data folder exsits, and that it cotains the following files:
    profiles: a folder containing the ASHP profiles
    UK_residential_population_2011_1_km.asc: a file containing the population grid data

    """

    def __init__(
        self,
        datapathway,
        weatherdatapathway,
        ASHPprofilepath="profiles/",
        populationsitelocs=None,
        desiredpopulation=61389177,
        yearmin=2010,
        yearmax=2010,
        DHWtemp=50,
    ):
        """This function initialises the demand class, and loads the data required to generate the demand time series
        datapathway: str: the pathway to the demand data
        weatherdatapathway: str: the pathway to the weather data
        ASHPprofilepath: str: the pathway to the ASHP profiles
        populationgridfile: str: the pathway to the population grid file. If None, the population grid file will be generated, and saved in the demand data folder
                                                                this may involve overwriting the existing population grid file
        desiredpopulation: int: the desired population to scale the population grid to
        """
        self.datapathway = datapathway
        self.weatherdatapathway = weatherdatapathway
        self.weathersitelocs = np.loadtxt(
            weatherdatapathway + "/site_locs.csv", delimiter=",", skiprows=1
        )
        self.ASHPprofilepath = ASHPprofilepath
        self.populationsitelocs = populationsitelocs
        self.startdatetime = datetime.datetime(yearmin, 1, 1)
        self.enddatetime = datetime.datetime(yearmax + 1, 1, 1)

        self.DHWtemp = DHWtemp
        if self.populationsitelocs == None:
            self.generatepopulationsitelocs()
        else:
            self.popsitelocs = np.loadtxt(
                f"{self.datapathway}/{self.populationsitelocs}",
                delimiter=",",
                skiprows=1,
            )
        currentpopulation = np.sum(self.popsitelocs[:, 3])
        scalingfactor = currentpopulation / desiredpopulation
        self.popsitelocs[:, 3] = self.popsitelocs[:, 3] / scalingfactor
        self.generatedailydemand()
        self.generateASHPdemand()

    def generatedailydemand(self):
        """This function calculates the daily degree days, BAIT and COPs for each site location, and then generates the daily demand time series.
        A population weighted estimate is used.
        """
        self.datadict = {}
        startdate = None
        hdds = []
        dhwCOPs = []
        heatingCOPs = []
        alltemps = []
        popsums = []

        for i in range(len(self.popsitelocs)):
            if self.popsitelocs[i, 3] == 0:
                continue
            popsums.append(self.popsitelocs[i, 3])
            code = int(self.popsitelocs[i, 0])
            if startdate == None:
                with open(f"{self.weatherdatapathway}/{code}.csv") as f:
                    data = f.read().strip().split("\n")
                    startdate = datetime.datetime.strptime(
                        data[1].split(",")[0], "%Y-%m-%d %H:%M:%S"
                    )
                startindex = int(
                    (self.startdatetime - startdate).total_seconds() // 3600
                )
                endindex = int((self.enddatetime - startdate).total_seconds() // 3600)
                print(f"Start index: {startindex}, End index: {endindex}")
            data = np.loadtxt(
                f"{self.weatherdatapathway}/{code}.csv",
                delimiter=",",
                skiprows=1,
                usecols=(1, 2, 3, 4),
            )[startindex:endindex]

            daydata = []
            for i in range(data.shape[1]):
                dayreshape = data[:, i].reshape(-1, 24)
                daydata.append(np.mean(dayreshape, axis=1))

            solar, temps, humid, wind = daydata
            temps = temps - 273.15  # convert to celsius
            dataframe = pd.DataFrame(
                {
                    "temperature": temps,
                    "humidity": humid,
                    "wind_speed_2m": wind,
                    "radiation_global_horizontal": solar,
                }
            )
            # convert temperature to celsius
            # convert humidity to g/kg
            dataframe["humidity"] = dataframe["humidity"] * 1000

            baits = demand_ninja.core._bait(dataframe, 0.5, 0.01, -0.18, 0.051)
            baits = baits.to_numpy()
            hdd = 13.2 - baits
            hdd[hdd < 0] = 0
            hdds.append(hdd)
            heatsinktemp = 40 - 1 * (dataframe["temperature"])
            heatingtempdelta = heatsinktemp - (dataframe["temperature"])
            DHWtempdelta = self.DHWtemp - (dataframe["temperature"])
            heatingtempdelta[heatingtempdelta < 15] = 15
            DHWtempdelta[DHWtempdelta < 15] = 15
            heatingCOP = 6.08 - 0.09 * heatingtempdelta + 0.0005 * heatingtempdelta**2
            dhwCOP = 3.5 - 0.05 * DHWtempdelta + 0.0005 * DHWtempdelta**2
            heatingCOPs.append(heatingCOP)
            dhwCOPs.append(dhwCOP)
            alltemps.append(dataframe["temperature"])

        # for every day, we average the values, using a population weighted average

        self.averagedhdds = np.average(hdds, axis=0, weights=popsums)
        self.averagedheatingCOPs = np.average(heatingCOPs, axis=0, weights=popsums)
        self.averageddhwCOPs = np.average(dhwCOPs, axis=0, weights=popsums)
        self.averagedtemps = np.average(alltemps, axis=0, weights=popsums)

        heatingpower = 140  # from demand ninja paper
        self.pbase = 20  # from analysis of base gas demand
        # the predicted demand is in GW, and we want to convert it to GWh, so we multiply by 24
        self.dailyvariableheatingdemand = [
            np.sum(popsums) * i * 24 * heatingpower / (10**9) for i in self.averagedhdds
        ]
        self.flatDHWdemand = (
            24 * self.pbase * 0.45
        )  # based on finding that DHW demand is 45% of base demand, from other analysis

    def generateASHPdemand(self):
        # hourly DHW profiles derived from the work of Watson
        DHWprofile = np.loadtxt(f"{self.ASHPprofilepath}/hourlyDHW.csv", delimiter=",")
        ASHPprofiles = np.loadtxt(
            f"{self.ASHPprofilepath}/hourlyASHP.csv", delimiter=",", skiprows=1
        )
        # ASHPs are used differently at different temperatures, so Watson's data has
        # a range of different profiles

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

        # we're using population weighted averages. Future work might work on a regional basis

        # variable heating demand, COPs and temps should all be the same length
        totaldemandtimeseries = []
        dhwdemandtimeseries = []
        ASHPelecdemandtimeseries = []
        for i in range(len(self.averagedtemps)):
            temp = self.averagedtemps[i]
            heatingCOP = self.averagedheatingCOPs[i]
            dhwCOP = self.averageddhwCOPs[i]
            heatingdemand = self.dailyvariableheatingdemand[i]
            DHWdemand = self.flatDHWdemand
            for j, bounds in enumerate(heatbounds):
                if temp >= bounds[0] and temp < bounds[1]:
                    heatingprofile = ASHPprofiles[:, j]
                    break
            DHWdemand = (DHWdemand / dhwCOP) * DHWprofile
            heatingdemand = (heatingdemand / heatingCOP) * heatingprofile
            totaldemand = heatingdemand + DHWdemand
            totaldemandtimeseries.append(totaldemand)
            dhwdemandtimeseries.append(DHWdemand)
            ASHPelecdemandtimeseries.append(heatingdemand)
        totaldemandtimeseries = np.hstack(totaldemandtimeseries)
        dhwdemandtimeseries = np.hstack(dhwdemandtimeseries)
        ASHPelecdemandtimeseries = np.hstack(ASHPelecdemandtimeseries)
        self.totaldemandtimeseries = totaldemandtimeseries
        self.dhwdemandtimeseries = dhwdemandtimeseries
        self.ASHPelecdemandtimeseries = ASHPelecdemandtimeseries

    def generatepopulationsitelocs(self):
        with open(
            f"{self.datapathway}/UK_residential_population_2011_1_km.asc"
        ) as file:  #
            data = file.read().strip().split("\n")
            xllcorner = float(data[2].split(" ")[-1])
            yllcorner = float(data[3].split(" ")[-1])
            cellsize = float(data[4].split(" ")[-1])
            data = data[6:]
            data = [i.strip().split(" ") for i in data]
            numrows = len(data)
            numcols = len(data[0])
            # converts all data to ints
            data = [[int(j) for j in i] for i in data]
        data = np.array(data)
        # the value -9999 is used to represent no data, so we set it to 0
        data[data == -9999] = 0
        # the ascii data we've loaded in is in British National Grid coordinates, so we need to convert it to latitudes and longitudes
        # the ascii data format can be unintuitive. The xllcorner and yllcorner define the coordiantes of the bottom left corner
        # the data is then read in row by row, with the first row being the most northerly

        # the xvals range will give us the x coordinates of the centre of each cell (using the cellsize as the step, and offsetting by half a cellsize)
        xvals = np.arange(
            cellsize / 2 + xllcorner,
            cellsize / 2 + xllcorner + cellsize * numcols,
            cellsize,
        )

        # the yvals range will give us the y coordinates of the centre of each cell (using the cellsize as the step, and offsetting by half a cellsize)
        # the first row will have the highest y value, so we need to start at the top of the grid and work our way down, hence the negative increment
        yvals = np.arange(
            yllcorner + cellsize * numrows + cellsize / 2,
            yllcorner + cellsize / 2,
            -cellsize,
        )
        # we now use meshgrid to create a grid of x and y values
        # a meshgrid is a way of creating a grid of coordinates from two vectors of x and y values
        x, y = np.meshgrid(xvals, yvals)
        # our x and y coordinates are all in British National Grid, so we need to convert them to latitudes and longitudes
        # we first create a transformer object, which will allow us to convert between the two projections
        # The from_crs function takes the EPSG code of the input projection, and the EPSG code of the output projection
        transformer = proj.Transformer.from_crs(27700, 4326)

        # we can now use the transformer object to convert our x and y coordinates to latitudes and longitudes
        lats, lons = transformer.transform(x, y)

        # Northern Ireland has a different grid system, so we need to remove the population data from there
        northernirelandmask = (x < 200000) & (y > 450000) & (y < 620000)
        data[northernirelandmask] = 0

        # now we have our grid of population, but its on a 1km grid. We will now use  our weather data site_locs,
        # and sum the population within each weather grid square

        merralatcellsize = 0.5
        merralongcellsize = 0.625
        popsitelocs = []
        for i in range(len(self.weathersitelocs)):
            sitelocnum = int(self.weathersitelocs[i][0])
            lat = self.weathersitelocs[i][1]
            lon = self.weathersitelocs[i][2]
            # we create a mask to select the population data within the weather grid square
            latmask = (lats >= lat - merralatcellsize / 2) & (
                lats < lat + merralatcellsize / 2
            )
            lonmask = (lons >= lon - merralongcellsize / 2) & (
                lons < lon + merralongcellsize / 2
            )
            mask = latmask & lonmask
            # mask will be true for all points within the weather grid square
            pop = np.sum(data[mask])
            popsitelocs.append([sitelocnum, lat, lon, pop])

        with open(f"{self.datapathway}/populationsitelocs.csv", "w") as file:
            file.write("sitelocnum,lat,lon,pop\n")
            for i in popsitelocs:
                file.write(f"{i[0]},{i[1]},{i[2]},{i[3]}\n")
        self.popsitelocs = popsitelocs


# %%
if __name__ == "__main__":
    b = demand(
        "demanddata",
        "D:/solar-humid-temp-wind-comb",
        ASHPprofilepath="demanddata/profiles/",
        populationsitelocs="populationsitelocs.csv",
        yearmin=2013,
        yearmax=2019,
        DHWtemp=50,
    )

    # %%
    from scipy.optimize import curve_fit

    demanddata = np.loadtxt(
        "c:/Users/SA0011/Documents/data/demand.csv",
        delimiter=",",
        skiprows=1,
        usecols=2,
    )
    starttime = datetime.datetime(2009, 1, 1)
    selectionstart = datetime.datetime(2017, 1, 1)
    hoursbetween = int((selectionstart - starttime).total_seconds() // 3600)
    demanddata = demanddata[hoursbetween:]
    # the data is in hour long periods: we want a daily average, so we reshape
    # to a row of 24 hour periods, and then take the mean of each row
    demanddata = demanddata.reshape(-1, 24)
    demanddata = np.mean(demanddata, axis=1)
    # %%
    Hdds = b.averagedhdds
    # save the hdds as a numpy file
    np.save("hdds.npy", Hdds)
    # save the demand data
    np.save("demanddata.npy", demanddata)
    quit()
    # these are the daily degree days between 2013-2019: we want the
    # average daily degree days for 2017-2019
    Hdds = Hdds[-len(demanddata) :]

    print(len(Hdds), len(demanddata))
    # we need to make an array which is 0 on weekdays, and 1 on weekends
    # the first day of 2017 was a sunday
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

    def func(x, Pbase, Pheat, weekendfactor):
        return Pbase + Pheat * x + weekendfactor * weekenddirac

    popt, pcov = curve_fit(func, Hdds, demanddata)

    predictedvals = func(Hdds, *popt)
    # find correlation coefficient
    r = np.corrcoef(predictedvals, demanddata)[0, 1]
    print(r)
    # print(popt)
    plt.plot(demanddata, label="Actual demand")
    plt.plot(func(Hdds, *popt), label="Predicted demand")

    plt.xlabel("Days")
    plt.ylabel("Demand (MW)")
    # now we want to plot the days of the week on a scatter overlaying the data

    # the first day of 2017 was a sunday
    sundayindices = np.arange(0, len(demanddata), 7)
    mondayindices = np.arange(1, len(demanddata), 7)
    tuesdayindices = np.arange(2, len(demanddata), 7)
    wednesdayindices = np.arange(3, len(demanddata), 7)
    thursdayindices = np.arange(4, len(demanddata), 7)
    fridayindices = np.arange(5, len(demanddata), 7)
    saturdayindices = np.arange(6, len(demanddata), 7)

    sundaydata = demanddata[sundayindices]
    mondaydata = demanddata[mondayindices]
    tuesdaydata = demanddata[tuesdayindices]
    wednesdaydata = demanddata[wednesdayindices]
    thursdaydata = demanddata[thursdayindices]
    fridaydata = demanddata[fridayindices]
    saturdaydata = demanddata[saturdayindices]

    plt.scatter(sundayindices, sundaydata, label="Sunday")
    plt.scatter(mondayindices, mondaydata, label="Monday")
    plt.scatter(tuesdayindices, tuesdaydata, label="Tuesday")
    plt.scatter(wednesdayindices, wednesdaydata, label="Wednesday")
    plt.scatter(thursdayindices, thursdaydata, label="Thursday")
    plt.scatter(fridayindices, fridaydata, label="Friday")
    plt.scatter(saturdayindices, saturdaydata, label="Saturday")

    plt.legend()
    plt.title("Predicted vs Actual Daily Demand 2017-2019")
    plt.show()

    # %%

    averageelecdemand = (
        np.sum(loadeddemanddata) / (len(loadeddemanddata) / (24 * 365)) / 10**6
    )
    scalingfactor = averageelecdemand / averagedemandminusheating
    loadeddemanddata = loadeddemanddata / scalingfactor
    # %%
    combineddemand = totaldemandseries * 1000 + loadeddemanddata
    currentdate = datetime.datetime(2013, 1, 1)
    with open("2013-2019demand.csv", "w") as f:
        f.write("datetime,Demand(MW)\n")
        for i in combineddemand:
            datestring = currentdate.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{datestring},{i}\n")
            currentdate += datetime.timedelta(hours=1)
# %%
# plot first 4 days on top of each other
for i in range(4):
    daydata = b.totaldemandtimeseries[i * 24 : (i + 1) * 24]
    normeddaydata = daydata / np.sum(daydata)
    plt.plot(normeddaydata, label=f"Day {i+1}")
plt.legend()
plt.show()
# %%
