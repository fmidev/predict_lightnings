#!/bin/sh
#





# Prepare the Anaconda Python environment 
export PATH="/lustre/tmp/kamarain/miniconda/bin:$PATH"
export LD_LIBRARY_PATH="/lustre/tmp/kamarain/miniconda/lib"

# Assure proxies
export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080
export ftp_proxy=http://wwwproxy.fmi.fi:8080 


# Github directory
codir='/home/users/kamarain/lightnings/'

# Data input directory
indir='/ibrix/arch/climate/kamarain/ERAInterim/global/'

# Results directory
oudir='/lustre/tmp/kamarain/lightnings/'


cd $codir


# Download ERA-Interim data --- bgn

declare -a vars=('t850' 'pmsl' 'rh700' 'cape')

for var in "${vars[@]}"
do
   echo $var
   #python download_eraint_from_ecmwf.py $var $codir $indir $oudir &
done
wait

# Download ERA-Interim data --- end




# Process data and run modeling experiments --- bgn

#python preprocess_raw_data.py $codir $indir $oudir &
wait

python fit_models_sequent.py $codir $indir $oudir &
wait

python plot_results.py $codir $indir $oudir &
wait

# Process data and run modeling experiments --- end





