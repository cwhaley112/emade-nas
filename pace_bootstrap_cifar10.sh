#!/bin/bash

read -p 'This script will setup the necessary SQL infrastructure to run EMADE on PACE & downloads the CIFAR-10 dataset.
WARNING: This will overwrite your SQL database if it already exists. Do you want to continue? [yn] ' run_setup

if [[ $run_setup == n ]] ; then
    exit
fi

cd $HOME

(ps -u "$(whoami)" | grep mysql | awk '{print $1}' | xargs -r kill -9 &)

rm -rf scratch
mkdir -p scratch/db
cp emade/pace/setup/.my.cnf .my.cnf

sed -i "s/cburdell3/$(whoami)/g" .my.cnf

let "port_number = 3300 + $RANDOM % 100"
sed -i "s/PORT_NUMBER/$port_number/g" .my.cnf

mysql_install_db --datadir=$HOME/scratch/db

cd /usr

echo 'What password do you want to use for your SQL database?'
read -s password

mysqld_safe --datadir="/storage/home/hpaceice1/$(whoami)/scratch/db" & { sleep 5 && cd $HOME && mysql -u root -e "DELETE FROM mysql.user WHERE user='';
    GRANT ALL PRIVILEGES ON *.* TO '$(whoami)'@'%' IDENTIFIED BY '$password' WITH GRANT OPTION;
    FLUSH PRIVILEGES;
    create database cifar10;"; }

(ps -u "$(whoami)" | grep mysql | awk '{print $1}' | xargs -r kill -9 &)


read -p 'What is the name of your conda environment? ' conda_env_name

cd $HOME/emade
rm -rf ../tensorflow_datasets
rm datasets/cifar10/emade_test_cifar10_0.npz 
rm datasets/cifar10/emade_train_cifar10_0.npz
module load anaconda3/2020.02
conda activate "$conda_env_name"
cd datasets/cifar10
python gen_cifar10.py

echo "Done!"