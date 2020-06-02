#!/bin/bash
repo="http://users.uoa.gr/~sdi1500071/eestech"
speakers="speakers.tar.gz"
inp="input/myinput"

# Installing requirements in a virtual environment
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

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
mkdir $inp/no_unfreezing
mv pretraining model_state.pth no_unfreezing.cfg $inp/no_unfreezing
mv data $inp
rm eestech
echo -e "\e[1;92mdone\e[0m"

# Predict and Evaluate
cd working && python3 predict.py && python3 eval.py && cd ..

# Clean up and exit
rm -f $inp/no_unfreezing/experiment.cfg $speakers
exit 0
