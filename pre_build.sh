#!/bin/bash
apt-get update
apt-get install -y gfortran gcc g++ libopenblas-dev liblapack-dev
apt-get clean
rm -rf /var/lib/apt/lists/*