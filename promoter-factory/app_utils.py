import os
import streamlit as st
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed
from Promoter_Calculator_v1_0 import Promoter_Calculator
# from stqdm import stqdm
import json
import subprocess
from glob import glob

def create_factory_root(factory_name):
    user_cache = os.path.expanduser("~/.cache")
    factory_root = os.path.join(user_cache, "promoter_factory", factory_name)
    os.makedirs(factory_root, exist_ok=True)
    return factory_root

def update_state_in_sidebar(session_state):
    with st.sidebar.expander("Session State", expanded=True):
        for key, value in session_state.items():
            if key == "factory_root":
                if value is None:
                    st.sidebar.markdown(f"**{key}**: None")
                else:
                    st.sidebar.markdown(f"**{key}**:\n{value.replace(os.path.expanduser('~'), '~')}")
            elif type(value) != dict:
                if value is None or session_state['factory_root'] is None:
                    st.sidebar.markdown(f"**{key}**: None")
                else:
                    st.sidebar.markdown(f"**{key}**:\n{value.replace(session_state['factory_root'], '')}")
            elif type(value) == dict:
                st.sidebar.markdown(f"**{key}**: ")
                df = pd.DataFrame(value.items(), columns=["Key", "Value"], dtype=str)
                st.sidebar.write(df)

def run_prokka(session_state):
    genome_file = session_state['genome_file']
    os.makedirs(f"{session_state['factory_root']}/prokka_output", exist_ok=True)
    cwd = os.getcwd()
    os.chdir(f"{session_state['factory_root']}/prokka_output")
    cmd = f"prokka --kingdom Bacteria {genome_file} --noanno --force"
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chdir(cwd)

def extract_upstream_sequences(session_state):
    prokka_gbk_file = glob(f"{session_state['factory_root']}/prokka_output/PROKKA_*/*.gbk")[0]
    putative_promoters = []
    with open(prokka_gbk_file) as input_handle:
        for record in SeqIO.parse(input_handle, "genbank"):
            seq = record.seq
            for feature in record.features:
                type = feature.type
                upper_seq_160bp = None
                if type == "CDS":
                    if feature.location.strand == 1 and feature.location.start > 160:
                        upper_seq_160bp = record.seq[feature.location.start-160:feature.location.start]
                    if feature.location.strand == -1 and feature.location.end < len(record.seq)-160:
                        upper_seq_160bp = record.seq[feature.location.end:feature.location.end+160]
                        upper_seq_160bp = upper_seq_160bp.reverse_complement()
                    if upper_seq_160bp:
                        putative_promoters.append(upper_seq_160bp)
    with open(f"{session_state['factory_root']}/prokka_output/putative_promoters.fasta", "w") as output_handle:
        for i, sequence in enumerate(putative_promoters):
            output_handle.write(f">seq_{i}\n{sequence}\n")


def predict_promoters(session_state):
    promotercalculator_output = os.path.join(session_state['factory_root'], "promotercalculator_output")
    prokka_output = os.path.join(session_state['factory_root'], "prokka_output")
    os.makedirs(promotercalculator_output, exist_ok=True)
    putative_promoters = []
    with open(f"{prokka_output}/putative_promoters.fasta") as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            putative_promoters.append(str(record.seq))
    calc = Promoter_Calculator()
    def process_sequence(sequence):
        calc.run(sequence, TSS_range=[60, len(sequence)])
        output = calc.output()
        score_list = [result['Tx_rate'] for result in output['Forward_Predictions_per_TSS'].values()]
        index = score_list.index(max(score_list))
        core_promoter = sequence[index:index + 60]
        return sequence, core_promoter, max(score_list)
    results = Parallel(n_jobs=-1)(delayed(process_sequence)(sequence) for sequence in putative_promoters)
    core_promoters = [result[1] for result in results]
    predicted_Tx_rates = [result[2] for result in results]
    df = pd.DataFrame({'Putative_Promoter': [result[0] for result in results], 'Core_Promoter': core_promoters, 'Predicted_Tx_Rate': predicted_Tx_rates})
    df.to_csv(f"{promotercalculator_output}/predicted_promoters.csv", index=False)
    with open(f"{promotercalculator_output}/predicted_promoters_strong.fasta", "w") as output_handle:
        for index, row in df.sort_values("Predicted_Tx_Rate",ascending=False).head(len(df)//5).iterrows():
            output_handle.write(f">{index}|ptx={row['Predicted_Tx_Rate']}\n{row['Core_Promoter']}\n")
    return f"{promotercalculator_output}/predicted_promoters_strong.fasta"

def save_session_state(session_state):
    if "factory_name" in session_state and session_state['factory_name'] is not None:
        factory_root = session_state['factory_root']
        with open(f"{factory_root}/session_state.json", "w") as output_handle:
            json.dump({k:session_state[k] for k in session_state}, output_handle)

def check_session_state(factory_name):
    user_cache = os.path.expanduser("~/.cache")
    factory_root = os.path.join(user_cache, "promoter_factory", factory_name)
    if os.path.exists(f"{factory_root}/session_state.json"):
        with open(f"{factory_root}/session_state.json") as input_handle:
            session_state = json.load(input_handle)
        return session_state
    return None