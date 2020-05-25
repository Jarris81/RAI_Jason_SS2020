#!/bin/sh

cd /home/jason/git/robotics-course
git pull
git submodule update 
cd build
cmake .. -DPYTHON_EXECUTABLE=/home/jason/anaconda3/envs/rai/bin/python -DPYTHON_LIBRARY=/home/jason/anaconda3/envs/rai/lib/libpython3.8.so
make -j4
