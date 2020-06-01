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

function fetch(){
    dir=$(dirname $1)
    base=$(basename $1)
    url=$2/$base
    echo -en "fetching \e[1;92m$dir\e[0m..."
    cd $dir && wget -r -np -nH --cut-dirs=1 $url -R "index.html*" &> /dev/null
    echo -e "\e[1;92mdone\e[0m"
    cd ..
}

# Fetch files necessary
dirs=("input/data"  "no_unfreezing/pretraining")

for dir in ${dirs[@]}
do
    if [ ! -e input/$dir ]; then fetch $dir $repo ; fi
done

# Fetch model state
cd no_unfreezing && wget $repo/model_state.pth && cd ..

# Predict and Evaluate
python3 predict.py
python3 eval.py
