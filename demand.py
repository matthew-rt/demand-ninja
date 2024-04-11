import numpy as np
import demand_ninja
import pyproj as proj
import datetime
import pandas as pd


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
        populationgridfile=None,
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
            weatherdatapathway + "sitelocs.csv", delimiter=",", skiprows=1
        )
        self.ASHPprofilepath = ASHPprofilepath
        self.populationgridfile = populationgridfile
        self.startdatetime = datetime.datetime(yearmin, 1, 1)
        self.enddatetime = datetime.datetime(yearmax + 1, 1, 1) - datetime.timedelta(
            hours=1
        )
        self.DHWtemp = DHWtemp
        if self.populationgridfile == None:
            self.generatepopulationsitelocs()
        else:
            self.popsitelocs = np.loadtxt(
                f"{self.datapathway}/{self.populationgridfile}",
                delimiter=",",
                skiprows=1,
            )
        currentpopulation = np.sum(self.popsitelocs[:, 3])
        scalingfactor = currentpopulation / desiredpopulation
        self.popsitelocs[:, 3] = self.popsitelocs[:, 3] / scalingfactor

    def generatedailydemand(self):
        """This function calculates the daily degree days, BAIT and COPs for each site location, and then generates the daily demand time series.
        A population weighted estimate is used.
        """
        self.datadict = {}
        startdate = None
        hdds = []
        dhwCOPs = []
        heatingCOPs = []
        temps = []
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
                        data[1].split(",")[0], "%d/%m/%Y %H:%M"
                    )
                startindex = (startdate - self.startdatetime).days * 24
                endindex = (self.enddatetime - startdate).days * 24

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

            temps, humid, wind, solar = daydata
            dataframe = pd.DataFrame(
                {
                    "temperature": temps,
                    "humidity": humid,
                    "wind_speed_2m": wind,
                    "radiation_global_horizontal": solar,
                }
            )
            # convert temperature to celsius
            dataframe["temperature"] = dataframe["temperature"] - 273.15
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
            temps.append(dataframe["temperature"])

        # for every day, we average the values, using a population weighted average

        self.averagedhdds = np.average(hdds, axis=0, weights=popsums)
        self.averagedheatingCOPs = np.average(heatingCOPs, axis=0, weights=popsums)
        self.averageddhwCOPs = np.average(dhwCOPs, axis=0, weights=popsums)
        self.averagedtemps = np.average(temps, axis=0, weights=popsums)

        heatingpower = 140  # from demand ninja paper
        self.pbase = 20  # from analysis of base gas demand
        # the predicted demand is in GW, and we want to convert it to GWh, so we multiply by 24
        self.dailyvariableheatingdemand = [
            np.sum(popsums) * i * heatingpower / (10**9) for i in self.averagedhdds
        ]
        self.flatDHWdemand = (
            24 * 20 * 0.45
        )  # based on finding that DHW demand is 45% of base demand, from other analysis

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
