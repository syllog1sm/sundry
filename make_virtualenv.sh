#!/usr/bin/env bash

cd /tmp
#wget http://www.python.org/ftp/python/2.7.5/Python-2.7.5.tgz /tmp
#tar -zxvf Python-2.7.5.tgz
#cd Python-2.7.5
#mkdir ~/.localpython
#./configure --prefix=$HOME/.localpython
#make
#make install
#cd /tmp
wget --no-check-certificate https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.5.2    .tar.gz
tar -zxvf virtualenv-1.5.2.tar.gz
cd virtualenv-1.5.2/
python setup.py install
cd $HOME
python /tmp/virtualenv-1.5.2/virtualenv.py ve
source ~/rsve/bin/activate
