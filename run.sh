#!/bin/bash
repo="http://users.uoa.gr/~sdi1500071/eestech"
speakers="speakers.tar.gz"
no_unfreezing="working/no_unfreezing"
inp="input"

# Extract speakers payload
if [ ! -e $speakers ]
then
    echo "Error: download $speakers first"
    exit 1
fi
echo " > Extracing speakers from $speakers..."
tar -C $inp -xzf $speakers

# Fetch data & model state
echo -en "Fetching data..."
wget -r -np -nH --cut-dirs=2 $repo -R "index.html*" &> /dev/null
mv pretraining model_state.pth $no_unfreezing
mv data $inp
rm eestech
echo -e "\e[1;92mdone\e[0m"

# Predict and Evaluate
cd working && python3 predict.py && python3 eval.py && cd ..

# Clean up and exit
rm -f working/no_unfreezing/experiment.cfg $speakers
exit 0