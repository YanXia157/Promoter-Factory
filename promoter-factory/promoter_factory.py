import os
import subprocess
from Bio import SeqIO
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, pipeline, DataCollatorForLanguageModeling
import torch
from datasets import Dataset
from joblib import Parallel, delayed
from Promoter_Calculator_v1_0 import Promoter_Calculator
import itertools
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import logging
import shutil
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PromoterAnnotationPipeline:
    def __init__(self, factory_name:str, data_path:str=None):
        self.factory_name = factory_name
        self.create_factory_root(data_path)


    def create_factory_root(self, data_path:str=None):
        if data_path is not None:
            factory_root = os.path.join(data_path, self.factory_name)
            os.makedirs(factory_root, exist_ok=True)
            self.factory_root = factory_root
        else:
            user_cache = os.path.expanduser("~/.cache")
            factory_root = os.path.join(user_cache, "promoter_factory", self.factory_name)
            os.makedirs(factory_root, exist_ok=True)
            self.factory_root = factory_root

    def run_prokka(self, genome_file):
        genome_path = os.path.join(self.factory_root, "genome.fna")
        shutil.copyfile(genome_file, genome_path)
        cmd = f"prokka --kingdom Bacteria {genome_path} --noanno --outdir {self.factory_root}/prokka_output 1> {self.factory_root}/prokka.log 2> {self.factory_root}/prokka_error.log"
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)

    def extract_upstream_sequences(self):
        prokka_gbk_file = glob(f"{self.factory_root}/prokka_output/*.gbk")[0]
        putative_promoters = []
        with open(prokka_gbk_file, "r") as input_handle:
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
        with open(f"{self.factory_root}/prokka_output/putative_promoters.fasta", "w") as output_handle:
            for i, sequence in enumerate(putative_promoters):
                output_handle.write(f">seq_{i}\n{sequence}\n")
    
    def predict_promoters(self):
        os.makedirs(f"{self.factory_root}/promotercalculator_output", exist_ok=True)
        putative_promoters = []
        with open(f"{self.factory_root}/prokka_output/putative_promoters.fasta") as input_handle:
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
        results = Parallel(n_jobs=-1)(delayed(process_sequence)(sequence) for sequence in putative_promoters if "N" not in sequence)
        core_promoters = [result[1] for result in results]
        predicted_Tx_rates = [result[2] for result in results]
        df = pd.DataFrame({'Putative_Promoter': [result[0] for result in results], 'Core_Promoter': core_promoters, 'Predicted_Tx_Rate': predicted_Tx_rates})
        df.to_csv(f"{self.factory_root}/promotercalculator_output/predicted_promoters.csv", index=False)
        with open(f"{self.factory_root}/promotercalculator_output/predicted_promoters_strong.fasta", "w") as output_handle:
            for index, row in df.sort_values("Predicted_Tx_Rate",ascending=False).head(len(df)//5).iterrows():
                output_handle.write(f">{index}|ptx={row['Predicted_Tx_Rate']}\n{row['Core_Promoter']}\n")
    
    def process(self, genome_file):
        logging.info("Running Prokka")
        self.run_prokka(genome_file)
        logging.info("Extracting upstream sequences")
        self.extract_upstream_sequences()
        logging.info("Predicting promoters")
        self.predict_promoters()


def convert_fasta_to_csv(fasta_path, tag="<|usp|>"):
    fasta_dict = {}
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()
                fasta_dict[header] = ""
            else:
                fasta_dict[header] += line.strip()
    names = []
    seqs = []
    tags = []
    ids = []
    for k, v in fasta_dict.items():
        names.append(k)
        seqs.append(v)
        tags.append(tag)
        ids.append("|f")
        names.append(k)
        seqs.append(v[::-1])
        tags.append(tag)
        ids.append("|r")
    data = {"ID": ids, "Name": names, "Seq": seqs, "Tag": tags}
    df = pd.DataFrame(data)
    df.to_csv(fasta_path + "_traintext.csv", index=False)
    return df


def run_prokka(genome_file):
    os.makedirs("prokka_output", exist_ok=True)
    genome_path = os.path.join("prokka_output", "genome.fna")
    with open(genome_path, "wb") as f:
        f.write(genome_file.read())
    os.chdir("prokka_output")
    cmd = "prokka --kingdom Bacteria genome.fna --noanno"
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    os.chdir("..")

def extract_upstream_sequences():
    prokka_gbk_file = glob("prokka_output/PROKKA_*/*.gbk")[0]
    putative_promoters = []
    with open(prokka_gbk_file) as input_handle:
        for record in SeqIO.parse(input_handle, "genbank"):
            seq = record.seq
            for feature in record.features:
                type = feature.type
                upper_seq_160bp = None
                if type == "CDS":
                    if feature.strand == 1 and feature.location.start > 160:
                        upper_seq_160bp = record.seq[feature.location.start-160:feature.location.start]
                    if feature.location.strand == -1 and feature.location.end < len(record.seq)-160:
                        upper_seq_160bp = record.seq[feature.location.end:feature.location.end+160]
                        upper_seq_160bp = upper_seq_160bp.reverse_complement()
                    if upper_seq_160bp:
                        putative_promoters.append(upper_seq_160bp)
    with open("prokka_output/putative_promoters.fasta", "w") as output_handle:
        for i, sequence in enumerate(putative_promoters):
            output_handle.write(f">seq_{i}\n{sequence}\n")

def predict_promoters():
    os.makedirs("promotercalculator_output", exist_ok=True)
    putative_promoters = []
    with open("prokka_output/putative_promoters.fasta") as input_handle:
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
    df.to_csv("promotercalculator_output/predicted_promoters.csv", index=False)
    with open("promotercalculator_output/predicted_promoters_strong.fasta", "w") as output_handle:
        for index, row in df.sort_values("Predicted_Tx_Rate",ascending=False).head(len(df)//5).iterrows():
            output_handle.write(f">{index}|ptx={row['Predicted_Tx_Rate']}\n{row['Core_Promoter']}\n")

def train_model(
        seed=42, 
        batch_size=32, 
        learning_rate=5e-5, 
        warmup_steps=1000, 
        logging_steps=1000, 
        save_steps=1000, 
        eval_steps=1000, 
        max_steps=10000, 
        fp16=True, 
        special_token="<|usp|>", 
        base_model_name="jinyuan22/promogen2-xsmall",
        fasta_data_path="promotercalculator_output/predicted_promoters_strong.fasta",
        output_dir="examples/promogen2_small_usp",
        device="cuda"
        ):
    # base_model_name = "jinyuan22/promogen2-small"
    # fasta_data_path = "promotercalculator_output/predicted_promoters_strong.fasta"
    # output_dir = "examples/promogen2_small_usp"
    torch.manual_seed(seed)
    model = GPT2LMHeadModel.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    model.resize_token_embeddings(len(tokenizer))
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    usp_df = convert_fasta_to_csv(fasta_data_path, tag=special_token)
    total_step = min([max_steps, int(len(usp_df) / batch_size) * 10])
    train_ds = Dataset.from_pandas(usp_df)
    def preprocess_function(samples):
        processed_samples = {"input_ids": [], "attention_mask": []}
        for i, dna_sequence in enumerate((samples["Seq"])):
            tag = samples["Tag"][i]
            if samples["ID"][i] == "|f":
                tokenized_input = tokenizer(f"{tag}5{dna_sequence}3{tag}", padding="longest", truncation=True, max_length=204)
            if samples["ID"][i] == "|r":
                tokenized_input = tokenizer(f"{tag}3{dna_sequence}5{tag}", padding="longest", truncation=True, max_length=204)
            processed_samples["input_ids"].append(tokenized_input["input_ids"])
            processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        return processed_samples
    tokenized_ds = train_ds.map(preprocess_function, batched=True, num_proc=8)
    train_testvalid = tokenized_ds.train_test_split(test_size=0.1)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=f"{output_dir}_output",
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        weight_decay=0.1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        max_steps=total_step,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        logging_strategy="steps",
        save_steps=save_steps,
        report_to="tensorboard",
        fp16=fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_testvalid["train"],
        eval_dataset=train_testvalid["test"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # st.write(trainer.evaluate())
    return trainer.evaluate(), output_dir

def generate_promoters(special_token, 
                       model_path, 
                       num_return_sequences=128*50, 
                       batch_size=256, 
                       max_new_tokens=60, 
                       repetition_penalty=1.0, 
                       top_p=1.0, 
                       temperature=1.0, 
                       device="cuda",
                       do_sample=True,
                       output_dir=None):
    # model_path = "examples/promogen2_small_usp"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    pipe = pipeline("text-generation", model=model, device=device, tokenizer=tokenizer)
    txt = f"{special_token}5"
    tag = special_token
    # num_return_sequences = 128 * 50
    # batch_size = 256
    # max_new_tokens = 60
    # repetition_penalty = 1.0
    # top_p = 1.0
    # temperature = 1.0
    do_sample = True
    all_outputs = []
    if num_return_sequences <= batch_size:
        outputs = pipe(txt, 
                       num_return_sequences=num_return_sequences,
                       max_new_tokens=max_new_tokens,
                       repetition_penalty=repetition_penalty,
                       top_p=top_p,
                       temperature=temperature,
                       do_sample=do_sample)
        all_outputs.extend(outputs)
    else:
        for i in range(0, num_return_sequences, batch_size):
            outputs = pipe(txt, 
                        num_return_sequences=batch_size,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        temperature=temperature,
                        do_sample=do_sample)
            all_outputs.extend(outputs)
    seqs = [output["generated_text"].replace("", "").replace("5", "").replace("3", "").replace(tag, "") for output in all_outputs]
    def score(seq, tag, device="cuda:0"):
        if tag == 'none':
            inputs = tokenizer(f"5{seq}3", return_tensors="pt")
        else:
            inputs = tokenizer(f"{tag}5{seq}3{tag}", return_tensors="pt")
        inputs.to(device)
        input_ids  = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return pred['loss'].item()
    scores = [score(seq, tag) for seq in seqs]
    sp_name = special_token.replace("<|", "").replace("|>", "")
    if output_dir is None:
        output_file_name = f"examples/{sp_name}_t_{temperature}_r_{repetition_penalty}_p_{top_p}.fasta"
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = f"{output_dir}/{sp_name}_t_{temperature}_r_{repetition_penalty}_p_{top_p}.fasta"
    with open(output_file_name, "w") as f:
        for i, (seq, score) in enumerate(zip(seqs, scores)):
            f.write(f">{i}|score={score}\n{seq}\n")
    return output_file_name

def analyze_kmers(native_fasta_file,
                  generate_promoter_file,
                  k=6):
    def fasta_to_list(fasta_file):
        fasta_seqs = []
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    header = line[1:]
                    fasta_seqs.append("")
                else:
                    fasta_seqs[-1] += line
        return fasta_seqs
    def generate_all_possible_kmers(k):
        bases = ['A', 'T', 'C', 'G']
        return [''.join(p) for p in itertools.product(bases, repeat=k)]
    def count_kmers_in_sequences(sequences, k=6):
        all_kmers = generate_all_possible_kmers(k)
        kmers_count = {kmer: 0 for kmer in all_kmers}
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if kmer in kmers_count:
                    kmers_count[kmer] += 1
        return kmers_count
    # native_fasta_file = "promotercalculator_output/predicted_promoters_strong.fasta"
    native_promoters = fasta_to_list(native_fasta_file)
    native_promoters_kmer_count = count_kmers_in_sequences(native_promoters)
    # generate_promoter_file = "examples/generated_promoters.fasta"
    generate_promoters = fasta_to_list(generate_promoter_file)
    generate_promoters = [s for s in generate_promoters if len(s) == 60]
    generate_promoters_kmer_count = count_kmers_in_sequences(generate_promoters[:len(native_promoters)])
    rho = spearmanr(list(native_promoters_kmer_count.values()), list(generate_promoters_kmer_count.values()))[0]
    prs = pearsonr(list(native_promoters_kmer_count.values()), list(generate_promoters_kmer_count.values()))[0]
    plt.figure(figsize=(5,5))
    sns.regplot(x=np.array(list(native_promoters_kmer_count.values()))/np.sum(list(native_promoters_kmer_count.values())), 
                y=np.array(list(generate_promoters_kmer_count.values()))/np.sum(list(native_promoters_kmer_count.values())),
                scatter_kws={'s': 4}, 
                label=f"Spearman correlation: {rho:.4f}\nPearson correlation: {prs:.4f}")
    plt.xlabel("Native promoters")
    plt.ylabel("Generated promoters")
    plt.legend()
    plt.tight_layout()
    output_file_name = generate_promoter_file.replace(".fasta", f"_kmer_analysis_k_{k}.png")
    plt.savefig(output_file_name)
    return output_file_name
    # st.pyplot()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("subcommand", type=str, choices=["promoter_annotation"], help="Subcommand to run")
    parser.add_argument("--factory_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--genome_file", type=str, default=None)
    
    args = parser.parse_args()
    if args.subcommand == "promoter_annotation":
        # args = parser.parse_args()
        pipeline = PromoterAnnotationPipeline(args.factory_name, args.data_path)
        pipeline.process(args.genome_file)