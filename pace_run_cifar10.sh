#!/bin/bash

bash cleanup.sh

read -p 'Run setup? [yn] ' run_setup

if [[ $run_setup == y ]] ; then
    rm -rf my_pace
    cp -r pace my_pace
    cp templates/input_cifar10.xml my_pace
    cd my_pace

    find ! -name 'run.sh' -type f -exec sed -i "s/cburdell3/$(whoami)/g" {} \;

    echo 'What is your PACE-ICE SQL password?'
    read -s password
    sed -i "s/PACE_SQL_PASSWORD/$password/g" input_cifar10.xml

    read -p 'What is the name of your conda environment? ' conda_env_name
    find ! -name 'run.sh' -type f -exec sed -i "s/CONDA_ENVIRONMENT_NAME/$conda_env_name/g" {} \;

    let "port_number = 3300 + $RANDOM % 100"
    sed -i "s/PORT_NUMBER/$port_number/g" setup/.my.cnf input_cifar10.xml pbsmysql.pbs

    cd ..

    cp my_pace/setup/.my.cnf ../.my.cnf 

    echo "Setup successful."
fi

echo "Launching EMADE."

qsub my_pace/pbsmysql.pbs
qsub my_pace/launchEMADE_cifar10.pbs

watch "qstat -n | grep "$(whoami)" | tail -10"

echo 'Done!'
