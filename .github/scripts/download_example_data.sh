#!/bin/sh
synda certificate renew
synda install CMIP6.ScenarioMIP.CCCma.CanESM5.ssp585.r3i1p2f1.Amon.tas.gn.v20190429 -y
synda daemon start
synda queue

echo sleep 10 and check again
sleep 10
synda queue
synda watch

example_file=$ST_HOME/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc
if [[ -f "$example_file" ]]; then
    echo "$example_file exists on your filesystem. Succes"
else
    # Did not find anything in the downloaded folder
    exit -1
fi
