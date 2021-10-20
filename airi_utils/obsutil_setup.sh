#!/bin/bash

INSTALLDIR=$HOME"/Documents"
cd $INSTALLDIR
echo "obsutil will be installed to: "$INSTALLDIR

if [ $(uname) == "Linux" ]; then
    DPATH="https://obs-community-intl.obs.ap-southeast-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz"
    SHELLCONFIG=$HOME"/.bashrc"

elif [ $(uname) == "Darwin" ]; then
    DPATH="https://obs-community-intl.obs.ap-southeast-1.myhuaweicloud.com/obsutil/current/obsutil_darwin_amd64.tar.gz"
    SHELLCONFIG=$HOME"/.zshrc"
fi

echo "installation file will be downloaded from: "$DPATH

FILE="$(basename -- $DPATH)"

mkdir -p obsutil

wget --quiet $DPATH -O $FILE

tar -xvf $FILE -C obsutil --strip-components 1

chmod 755 obsutil

# FILL HERE S3 OBS Server parameters: -i=Access Key Id, -k=Secret Access Key
obsutil/obsutil config -i=$1 -k=$2 -e=obs.ru-moscow-1.hc.sbercloud.ru

echo "shell config file " $SHELLCONFIG "will be modified"


SHELLCONFIG=$HOME"/Documents/text.txt"

if ! grep -q "obsutil config" $SHELLCONFIG ; then
    echo "shell config modified successfully"
    echo -e "\n\n# obsutil config \nalias obsutil=\""$INSTALLDIR"/obsutil/obsutil\"\n\n" >> $SHELLCONFIG

else
    echo "shell already configured, no changes required"
fi
