#!/bin/bash
repo="http://users.uoa.gr/~sdi1500071/eestech"
speakers="speakers.tar.gz"

# Extract speakers payload
if [ ! -e $speakers ]
then
    echo "Error: download $speakers first"
    exit 1
fi
echo " > Extracing speakers from $speakers..."
tar -C input -xzf $speakers
rm -f $speakers

# Fetch data
echo -en "Fetching data..."
wget -r -np -nH --cut-dirs=2 $repo -R "index.html*" &> /dev/null
mv pretraining model_state.pth no_unfreezing
mv data input
rm eestech
echo -e "\e[1;92mdone\e[0m"

# Fetch model state
cd no_unfreezing && wget -O model_state.pth $repo/model_state.pth && cd ..

# Predict and Evaluate
python3 predict.py
python3 eval.py
