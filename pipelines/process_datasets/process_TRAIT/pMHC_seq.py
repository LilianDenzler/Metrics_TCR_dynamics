import pandas as pd
import argparse
import os
import sys
import requests
from bs4 import BeautifulSoup
import numpy as np
import urllib.parse

human_beta2m_sequence="MSRSVALAVLALLSLSGLEAIQRTPKIQVYSREPAENGKPNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"

# Step 1: Fetch the accession code for a given HLA allele name
def generate_url(allele_name):
    base_url = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele"
    query = f'or(startsWith(name,"{allele_name}"),contains(previous_nomenclature,"{allele_name}"),eq(accession,"{allele_name}"))'
    params = {
        'limit': 1,
        'project': 'HLA',
        'fields': 'name,accession',
        'query': query
    }
    url = f"{base_url}?{urllib.parse.urlencode(params, safe='(),*')}"
    return url

def fetch_accession_code(allele_name):
    url = generate_url(allele_name)
    print(url)
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['accession']
        else:
            raise ValueError(f"No data found for allele {allele_name}")
    else:
        raise ValueError(f"Error fetching data for allele {allele_name}: {response.status_code}")

# Step 2: Fetch the full protein sequence using the accession code
def fetch_full_protein_sequence(accession_code):
    sequence_url = f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=imgthlapro;id={accession_code}&format=fasta&style=raw"
    response = requests.get(sequence_url)

    if response.status_code == 200:
        fasta_data = response.text
        # Remove the FASTA header and concatenate the sequence lines
        sequence = ''.join(fasta_data.split('\n')[1:])
        return sequence
    else:
        raise ValueError(f"Sequence for accession code {accession_code} not found.")

def get_seq(allele_name):
    try:# Fetch the accession code
        allele_name=allele_name.replace("HLA-","")
        accession_code = fetch_accession_code(allele_name)
        # Fetch the full protein sequence
        full_sequence = fetch_full_protein_sequence(accession_code)
        return full_sequence
    except ValueError as e:
        print(e)



def get_alleles(allele_name, loaded_df=None):
    if allele_name=="B2M":
        return human_beta2m_sequence

    # Extract data
    loaded_df = pd.read_csv(loaded_df) if loaded_df else None
    loaded_alleles = loaded_df['allele'].unique() if loaded_df is not None else []

    sequence_cache = {}
    if allele_name not in loaded_alleles:
        sequence = get_seq(allele_name)
        if sequence is None:
            print(f"Warning: sequence for allele {allele_name} not found. Using placeholder.")
            sequence = "UNKNOWNSEQUENCE!"
        sequence_cache[allele_name] = sequence
    else:
        sequence = loaded_df.loc[loaded_df['allele'] == allele_name, 'sequence'].values[0]
    trunc_seq=truncate(allele_name, sequence)
    return trunc_seq



def get_full_seq_no_encoding(input_df,out_file):
    data=[]
    allele_list,complex_ids,peptides,alleles=get_alleles(input_df)
    pmhc_output = []

    for complex_id, allele, peptide in zip(complex_ids, allele_list, peptides):
        try:
            if "UNKNOWN" in allele:
                continue
            else:
                data.append((int(complex_id), allele, peptide))
        except:
            print("allele ",allele, " could not be added")

    #save as dataframe
    df = pd.DataFrame(data, columns=['pdb_id', 'mhc', 'peptide'])
    df.to_csv(out_file, index=False)
    print(f"full sequences saved to {out_file}")
    return out_file

def truncate(MHC_name, sequence, truncation_path="MHC_TRUNCATION.csv"):
    truncation_df= pd.read_csv(truncation_path)
    MHC_class = MHC_name.split("*")[0]
    if MHC_class in truncation_df['MHC_name'].values:
        truncation_row = truncation_df[truncation_df['MHC_name'] == MHC_class]
        start = truncation_row['start'].values[0]
        end = truncation_row['end'].values[0]
        return sequence[start:end]
    else:
        print(f"No truncation data found for {MHC_name}. Returning full sequence.")
        return sequence

if __name__ == "__main__":
    #seq=get_alleles("HLA-B*08:01")
    seq=get_alleles("HLA-A*01:01")
    print("truncated sequence:", seq)

    """
    input_df = pd.read_csv(args.input)
    out_file=args.output
    get_full_seq_no_encoding(input_df, out_file)
    """
