import pandas as pd
import re
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import ntpath
import numpy as np
import argparse
import os
from itertools import islice

def main(params):
    input_dir = params.in_path
    output_dir = params.out_path

    os.makedirs(output_dir)
    bin_translate(os.path.join(input_dir,'bin_1.csv'), output_dir)
    bin_translate(os.path.join(input_dir,'bin_2.csv'), output_dir)
    bin_translate(os.path.join(input_dir,'bin_3.csv'), output_dir)
    bin_translate(os.path.join(input_dir,'bin_4.csv'), output_dir)
    bin_translate(os.path.join(input_dir,'bin_5.csv'), output_dir)

def bin_translate(bin_path, save_path, min_length_truncation=6):
    filename = ntpath.basename(bin_path)
    save_filename = filename.split('.csv')[0] + '_translated.csv'
    save_name = os.path.join(save_path, save_filename)

    print(f"Processing {save_name}")
    df = pd.read_csv(bin_path)
    df_escape = df.copy()
    df_escape.Sequences = df.Sequences.apply(escape_sequences)
    df2 = pd.DataFrame(df_escape.Sequences.tolist(), index=[df['Unnamed: 0'], df['Count'], df['Label']]).stack().reset_index(name='Sequences')[['Sequences', 'Count', 'Unnamed: 0', 'Label']]
    df3 = df2.copy()
    df3['Translated_sequences'] = df3.Sequences.apply(translate)
    df3['Protein_length'] = df3.Translated_sequences.apply(len)

    df3_filtered = df3[df3['Translated_sequences'].str.find('X') == -1]
    df3_filtered = df3_filtered[df3_filtered.Protein_length > min_length_truncation]
    k_mer_seq = df3_filtered.Sequences.apply(get_k_mers_str)
    df3_filtered['k_mer_sequences'] = k_mer_seq
    df3_filtered.to_csv(save_name)

    print(f"Saving {save_name}")
    print(f"Resulting bin length: {len(df3_filtered)}")

# escape from sequences all unnecessary characters and make a list of sequences
def escape_sequences(str):
    str = str.replace("'", "")
    str = str.replace("[", "")
    str = str.replace("]", "")
    str = str.replace(" ", "")
    return str.split(",")

def translate(sequence):
  sequence = Seq(sequence, generic_dna)
  protein = sequence.translate(to_stop=True)
  return protein.__str__()

def k_mers(sequence, k):
    j = 0
    it = iter(sequence)
    result = tuple(islice(it, k))
    if len(result) == k:
      yield "".join(result)
      for elem in it:
        result = result[1:] + (elem,)
        j += 1
        yield "".join(result)

def get_k_mers_str(sequence, k=3):
  gen = k_mers(sequence, k)
  gen_list = list(gen)
  k_mers_str = " ".join(gen_list)
  return k_mers_str



if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default='../data/bins/', help='input bins directory path', type=str)
    parser.add_argument('--out_path', default='../data/bins_translated', help='translated output bins directory path', type=str)
    params = parser.parse_args()

    main(params)

