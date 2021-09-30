#!/bin/bash

root_server='/share/catalyst/'
root_obs='obs://esg-korovin/'
file='data_mod1.lmdbz'
obs='/share/obsutil/obstil'
path='ocp_datasets/data/is2re/'

if [ "$2" == "-h" ]; then
  echo "Usage: `basename $0` copy file to obs server, parameters [up, down] [10k, all, val]"
  exit 0

elif [ "$2" == "10k" ]; then
  dataset='10k/train/'

elif [ "$2" == "all" ]; then
  dataset='all/train/'

elif [ "$2" == "val" ]; then
  dataset='all/val_ood_both/'

else
  echo 'path not specified, exit 0.\n'
  exit 0
fi

if [ "$1" == "up" ]; then
	echo 'copy' ${root_server}${path}${dataset}${file} to ${obs_server}${path}${dataset}
	${obs} cp ${root_server}${path}${dataset}${file} ${obs_server}${path}${dataset}

elif [ "$2" == "down" ]; then
	echo 'copy' ${obs_server}${path}${dataset}${file} to ${root_server}${path}${dataset}
	${obs} cp ${obs_server}${path}${dataset}${file} ${root_server}${path}${dataset}

else
  echo 'mode not specified, exit 0.\n'
  exit 0
fi
