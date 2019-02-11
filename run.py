from sstFronts import *

year = 2011
# month='02'

sst_dir = "../data/ODYSEA/odyssea_saf_extraction/"
bathy_path =  "../data/SRTM/SRTM30_0_360_new.nc"
sarpline= "../data/along-track_altimetry/Africana203_SARP.txt"

figs_dir = "../figs/"
results_dir = "../results/"
directories_list = [figs_dir, results_dir]

# run_all_years(sst_dir)

bathymetry = read_bathymetry(bathy_path)

make_directories(directories_list)
make_directories(year=year)
main(sst_dir, bathymetry, sarpline, year=year)

