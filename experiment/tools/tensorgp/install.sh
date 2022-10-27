# Remove `tensorgp` directory, if it exists.
if [ -d tensorgp ]; then
    rm -rf tensorgp
fi

# Clone TensorGP.
git clone https://github.com/AwardOfSky/TensorGP.git tensorgp
cd tensorgp

# Check out an appropriate commit.
git checkout 09e6d048c604fd979ad29b9c28a79ed919f41d18
# git checkout d75fb60a74f1965d9115bbef083e3762d76fa48c

# Update the core `engine.py` file within TensorGP 
# for the purposes of profiling.
\cp ../custom/engine.py ./tensorgp/engine.py