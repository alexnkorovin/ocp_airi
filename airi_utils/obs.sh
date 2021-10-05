#!/bin/bash

root_server='/share/catalyst/'
obs_server='obs://esg-korovin/'
file='data_mod1.lmdbz'
obs='/share/obsutil/obsutil'
path='ocp_datasets/data/is2re/'

# if [ "$2" == "-h" ]; then
#   echo "Usage: `basename $0` copy file to obs server, parameters [up, down] [10k, all, val]"
#   exit 0

# elif [ "$2" == "10k" ]; then
#   dataset='10k/train/'

# elif [ "$2" == "all" ]; then
#   dataset='all/train/'

# elif [ "$2" == "val" ]; then
#   dataset='all/val_ood_both/'

# else
#   echo 'path not specified, exit 0.\n'
#   exit 0
# fi

if [ "$1" == "up" ]; then
	echo 'copy' ${root_server}${path}$2$3 to ${obs_server}${path}$2
	${obs} cp ${root_server}${path}$2$3 ${obs_server}${path}$2

elif [ "$1" == "down" ]; then
	echo 'copy' ${obs_server}${path}$2$3 to ${root_server}${path}$2
	${obs} cp ${obs_server}${path}$2$3 ${root_server}${path}$2

else
  echo 'mode not specified, exit 0.\n'
  exit 0
fi
