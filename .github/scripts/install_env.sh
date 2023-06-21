#!/bin/sh
git clone https://github.com/JoranAngevaare/optim_esm_base.git ../base_install
cd ../base_install || bash install_software.sh $1

pip install -e ../optim_esm_tools
pip install -r requirements_tests.txt
