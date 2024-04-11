# %%
import numpy as np
import pyproj as proj
import time
import matplotlib.pyplot as plt


# first we need to load in our km grid dataset. It is in a folder called locationdata,
# and is made up of a .asc file and a .prj file. The .asc file is a text file with the
# data in it, and the .prj file is a projection file that tells us what projection the data is in.

with open("locationdata/UK_residential_population_2011_1_km.asc") as file:  #
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
data[data == -9999] = 0
print(np.max(data))
# now we need to make a grid of latitudes and longitudes

# we can use the pyproj library to convert between different projections
# we need to know the projection of the data, which is in the .prj file
# the projection of the data is British National Grid, which has an EPSG code of 27700
# we can use this code to create a projection object

# create a projection object for British National Grid

# create a grid of latitudes and longitudes using the projection object
xvals = np.arange(
    cellsize / 2 + xllcorner, cellsize / 2 + xllcorner + cellsize * numcols, cellsize
)
yvals = np.arange(
    yllcorner + cellsize * numrows + cellsize / 2, yllcorner + cellsize / 2, -cellsize
)
x, y = np.meshgrid(xvals, yvals)
transformer = proj.Transformer.from_crs(27700, 4326)
lats, lons = transformer.transform(x, y)

northernirelandmask = (x < 200000) & (y > 450000) & (y < 620000)
data[northernirelandmask] = 0
# %%
# convert the grid of latitudes and longitudes to British National Grid coordinates


# find index of the point in data with the highest value
maxindex = np.unravel_index(np.argmax(data, axis=None), data.shape)

# find the corresponding latitude and longitude
maxlat = lats[maxindex]
maxlon = lons[maxindex]


sitelocs = np.loadtxt(
    "C:/Users/SA0011/Documents/data/solar_comb/sitelocs.csv", delimiter=",", skiprows=1
)
latcellsize = 0.5
longcellsize = 0.625

# we're now going to iterate through the sitelocs, and then use a mask to select the
# lat and lon points whih are within each cell. We then want to plot this on a grid,
# which will update as we iterate through the sitelocs

popsitelocs = []
populationdisplaygrid = np.zeros_like(data)
for i in range(len(sitelocs)):
    sitelocnum = int(sitelocs[i][0])
    lat = sitelocs[i][1]
    lon = sitelocs[i][2]
    latmask = (lats >= lat - latcellsize / 2) & (lats < lat + latcellsize / 2)
    lonmask = (lons >= lon - longcellsize / 2) & (lons < lon + longcellsize / 2)
    mask = latmask & lonmask
    pop = np.sum(data[mask])
    populationdisplaygrid[mask] = pop
    popsitelocs.append([sitelocnum, lat, lon, pop])

with open("populationsitelocs.csv", "w") as file:
    file.write("Site number, Latitude, Longitude, Population\n")
    for i in popsitelocs:
        file.write(f"{i[0]},{np.round(i[1],3)},{np.round(i[2],3)},{i[3]}\n")

plt.imshow(populationdisplaygrid, cmap="hot")
plt.colorbar()
plt.show()

# %%
