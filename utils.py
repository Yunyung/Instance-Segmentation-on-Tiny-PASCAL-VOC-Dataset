import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil
from argparse import ArgumentParser

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def default_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment-file", help="Your training/testing configuration for the certain experiment")
    parser.add_argument("--resume", default="False", help="If resume train process")
    return  parser