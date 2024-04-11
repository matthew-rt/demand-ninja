import numpy as np
import demand_ninja


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
    ):
        """This function initialises the demand class, and loads the data required to generate the demand time series
        datapathway: str: the pathway to the demand data
        weatherdatapathway: str: the pathway to the weather data
        ASHPprofilepath: str: the pathway to the ASHP profiles
        populationgridfile: str: the pathway to the population grid file"""
        self.datapathway = datapathway
        self.weatherdatapathway = weatherdatapathway
        self.weathersitelocs = np.loadtxt(
            weatherdatapathway + "sitelocs.csv", delimiter=",", skiprows=1
        )
        self.ASHPprofilepath = ASHPprofilepath
        self.populationgridfile = populationgridfile

        if self.populationgridfile == None:
            self.generatepopulationsitelocs()

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
