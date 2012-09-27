#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -q gpu
#PBS -N output_test

cd $PBS_O_WORKDIR
./lab7
