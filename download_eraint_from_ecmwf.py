#!/usr/bin/env python
#
#

import os, sys
from ecmwfapi import ECMWFDataServer



# To run this example, you need an API key
# available from https://api.ecmwf.int/v1/key/

server = ECMWFDataServer()


# Variable key from command line arguments
varname = str(sys.argv[1]) 


# Directories from command line arguments
codir, indir, oudir = str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4])


name2code = {'pmsl': ["151",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'psfc': ["134",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'te2m': ["167",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'z500': ["129",        "an",   "pl",   "500",  "00/06/12/18",  "0"],
             'z70':  ["129",        "an",   "pl",   "70",   "00/06/12/18",  "0"],
             'vo850':["138",        "an",   "pl",   "850",  "00/06/12/18",  "0"],
             't850': ["130",        "an",   "pl",   "850",  "00/06/12/18",  "0"],
             'vo500':["138",        "an",   "pl",   "500",  "00/06/12/18",  "0"],
             'rh700':["157",        "an",   "pl",   "700",  "00/06/12/18",  "0"],
             'sst':  ["34",         "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'snw':  ["141",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'smo':  ["39/40/41",   "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'tcw':  ["136",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'cape': ["59",         "fc",   "sfc",  "0",    "00/12",        "03/06/09/12"],
             'prc':  ["228",        "fc",   "sfc",  "0",    "00/12",        "06/12"],
             'wg10': ["49",         "fc",   "sfc",  "0",    "00/12",        "06/12"],
             'ws10': ["207",        "an",   "sfc",  "0",    "00/06/12/18",  "0"],
             'aice': ["31",         "an",   "sfc",  "0",    "00/06/12/18",  "0"]}




for year in range(1979,2019):
    basename = "%s_eraint_1.125deg_%04d" % (varname,year)
    ncfile   = indir+"%s.nc" % (basename)
    if os.path.exists(ncfile):
        continue
    opts     = {
                "stream"    : "oper",
                "dataset"   : "interim",
                "class"     : "ei",
                "date"      : "%04d-01-01/to/%04d-12-31" % (year,year),
                "area"      : "europe",
                "grid"      : "0.75/0.75",
                "levelist"  : name2code[varname][3],
                "levtype"   : name2code[varname][2],
                "param"     : name2code[varname][0],
                "time"      : name2code[varname][4],
                "step"      : name2code[varname][5],
                "type"      : name2code[varname][1],
                "format"    : "netcdf",
                "target"    : ncfile
               }
    server.retrieve(opts)
    """
    # convert to netcdf and from spectral to gaussian
    if(name2code[varname][2]=="pl"):  
        os.system("cdo -P 4 -t ecmwf -r -f nc remapbil,r320x160 -sp2gpl "+" "+grbfile+" "+ncfile)

    if(name2code[varname][2]=="sfc" and varname!="smo"): 
        os.system("cdo -P 4 -t ecmwf -r -f nc remapbil,r320x160 -setgridtype,regular "+" "+grbfile+" "+ncfile)  

    if(name2code[varname][2]=="sfc" and varname=="smo"):
        os.system("cdo -P 4 -t ecmwf -r -f nc remapbil,r320x160 -setgridtype,regular "+" "+grbfile+" temp_file.nc")
        os.system("cdo -P 4 expr,'SMO=SWVL1+SWVL2+SWVL3' temp_file.nc "+ncfile); os.system("rm temp_file.nc")

    os.unlink(grbfile)
    """
print("Finished downloading ERA-Interim!")
