import json
import os
from pyqubo import Array, Constraint, Placeholder
import openjij as oj
import dimod
import time
import math
import numpy as np
import statistics as stats
import pandas as pd
import sys

from SQAVerifierOneshot import ImportImputJSON, AnnealingInfo, ExecuteQA

def getArgments():
    if len(sys.argv) != 4:
        print("python3 SQAVerifierXcaleNumsweeps.py 1stRange 2ndRange 3rdRange")
    else:
        args = []
        args.append(int(sys.argv[1]))
        args.append(int(sys.argv[2]))
        args.append(int(sys.argv[3]))
    return args

def ExportResult(Result, filename):
    df = pd.DataFrame.from_dict(Result)
    df = df.T
    path = f'output/{filename}.json'
    df.to_json(path)
    print(df)

def Main():
    edges, Group = ImportImputJSON()

    parameters = AnnealingInfo(edges, Group)

    ranges = getArgments()

    Result = {}
    for num_sweeps in range(ranges[0], ranges[1], ranges[2]):
        parameters.ChangeParameters(num_sweeps=num_sweeps)
        resultTable = ExecuteQA(edges, Group, parameters)
        Result[f'{num_sweeps}'] = resultTable 
        print(num_sweeps)

    ExportResult(Result, 'test') #'test'部分を保存したい名前に変更．できれば自動化したいな

if __name__ == "__main__":
    Main()
