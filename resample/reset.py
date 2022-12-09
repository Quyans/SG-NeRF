
from argparse import ArgumentParser
import os
import numpy as np
import shutil


def main():
    parser = ArgumentParser()

    parser.add_argument('--baseSrc', type=str, help='one out of: "train", "test"',
                        default="./data_src/scannet/scans"
    )

    parser.add_argument("--expname", type=str, default="scene0046_00sparse", \
        help='specify the experiment, required for "test" or to resume "train"')

    
    parser.add_argument("--sampleType",type=int,default=0,help="0shows using step,1shows using input xyz")
    parser.add_argument("--step",type=int,default=50,help="0shows using step,1shows using input xyz")