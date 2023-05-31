#!/bin/sh
git clone https://github.com/JoranAngevaare/optim_esm_base.git base_install
cd base_install
bash install_software.sh
cd ..
pip install -e ../optim_esm_tools
