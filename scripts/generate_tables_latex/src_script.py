#!/usr/bin/env python3
#-*- encoding: utf-8 -*-

import argparse
import copy
import datetime
import json
import yaml
import os
import sys
import time

from functools  import reduce
import pandas as pd

from sklearn.model_selection import ParameterGrid

# from tables_latex_util import create_example_table

def create_grid_search_table(params_dict: dict, params_list: list, filename: str) -> pd.DataFrame:
    
    records_list : list = list()

    result_reduce: int = reduce(
        lambda a,b: a*b,
        list(
            map(
                lambda item: len(item[1]),
                params_dict.items()
            )
        )
    )
    print()
    print(f"Number of Hyper-parameters combinations found: {result_reduce}")
    for ii, g in enumerate(ParameterGrid(params_dict)):
        record: list = list()
        for jj, item in enumerate(params_list):
            record.append(g[item])
            pass
        records_list.append(record)
        pass

    df = pd.DataFrame(data=records_list, columns=params_list)
    df.to_csv(f"{filename}",index=False)
    return copy.deepcopy(df)

def get_custom_parser():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help="Input file with specification for creating csv and latex tables.", type=str)
    parser.add_argument('--output_file', help="Output file specified for storing csv and latex tables.", type=str)
    params = parser.parse_args()

    return params, parser

def read_input_file(input_file: str) -> dict:
    with open(input_file) as f:
        if input_file.endswith('json') is True:
            result_dict = json.load(f)
        elif input_file.endswith('yaml') is True:
            result_dict = yaml.load(f)
        else:
            raise Exception(f"ERROR: file {input_file} has an extension not yet allowed!")
    return result_dict

def main(cmd_args, data_dict: dict):
    # create_example_table()
    tmp_data_dict: dict = copy.copy(data_dict)

    params_list: list = data_dict['params_list']
    del data_dict['params_list']

    print()
    print("Create csv file with comined hyper-parameters to be tested...")
    df: pd.DataFrame = create_grid_search_table(data_dict, params_list, cmd_args.output_file)

    from latex import build_pdf

    # pdf = build_pdf(min_latex)
    res = df.to_latex(index=False)
    min_latex = (r"\documentclass{article}"
             r"\begin{document}"
             # r"Hello, world!"
             r"\end{document}")

    pdf = build_pdf(min_latex)

    pdf.save_to('ex1.pdf')
    pass

if __name__ == "__main__":

    print()
    print("Parsing input params...")
    params, _ = get_custom_parser()

    print("Reading input file...")
    data_dict: dict = read_input_file(params.input_file)

    print("Running main part of script...")
    main(params, data_dict)

    pass