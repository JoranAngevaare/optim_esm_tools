#!/bin/sh
synda certificate renew
synda install CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r3i1p2f1.Amon.tas.gn.v20190429 -y
synda daemon start
synda queue

example_file=$ST_HOME/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc

for n_tries in {1..6}
do
    synda queue
    synda watch

    if [[ -f "$example_file" ]];
    then
        break
    fi

    echo "No file sleep $n_tries"

    if (( $n_tries > 4 ));
    then
        # By now we have waited 1+2+3+4 = 10 s, let's renew the certificate each time, just to be sure
        synda certificate renew
        synda daemon start
    fi
    sleep $n_tries
done


if [[ -f "$example_file" ]]; then
    echo "$example_file exists on your filesystem. Succes"
else
    # Did not find anything in the downloaded folder
    echo "perhaphs anything in the logs?"
    tail $ST_HOME/log/*
    exit -1
fi
