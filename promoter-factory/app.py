import streamlit as st
import json
from promoter_factory import train_model, generate_promoters, analyze_kmers
from names_generator import generate_name
from app_utils import create_factory_root, update_state_in_sidebar, run_prokka, extract_upstream_sequences, predict_promoters, save_session_state, check_session_state
# import subprocess
from glob import glob
import os
import pickle

st.set_page_config(page_title="Promoter Factory", page_icon="🧬", layout="wide")
st.title("Promoter Factory")

if "factory_root" not in st.session_state:
    st.session_state['factory_root'] = None

if "factory_name" not in st.session_state:
    st.session_state['factory_name'] = generate_name()

if "genome_file" not in st.session_state:
    st.session_state['genome_file'] = None

if "train_data" not in st.session_state:
    st.session_state['train_data'] = None

if "training_parameters" not in st.session_state:
    st.session_state['training_parameters'] = None

if "trained_model_path" not in st.session_state:
    st.session_state['trained_model_path'] = None

if "predicted_promoter_file" not in st.session_state:
    st.session_state['predicted_promoter_file'] = None

if "eval_output" not in st.session_state:
    st.session_state['eval_output'] = None

# try to collect generated data

st.markdown("### Let's create/load a factory for your chassis!")
columns_0 = st.columns([1,2])
with columns_0[0]:
    project_name = st.text_input("Factory Name", value=st.session_state['factory_name'])
    st.session_state['factory_name'] = project_name
    if st.button("Create or Load a Factory", use_container_width=True):
        factory_root = create_factory_root(project_name)
        st.session_state['factory_root'] = factory_root
        st.session_state['factory_name'] = project_name
# st.sidebar.header("Input Parameters")
if st.session_state['factory_root'] is None:
    st.error("Please create or load a factory.")
    st.stop()

st.sidebar.markdown("## Factory Information")
factory_root = create_factory_root(st.session_state['factory_name'])
st.session_state['factory_root'] = factory_root
prokka_files = f"{st.session_state['factory_root']}/prokka_output/putative_promoters.fasta"
genome_files = f"{st.session_state['factory_root']}/genome_file.fasta"
# st.write(genome_files)
predicted_promoter_files = f"{st.session_state['factory_root']}/promotercalculator_output/predicted_promoters_strong.fasta"
training_args = f"{st.session_state['factory_root']}/training_parameters.json"
uploaded_train_data = f"{st.session_state['factory_root']}/promoter_file.fasta"
if os.path.exists(genome_files):
    st.session_state['genome_file'] = genome_files
else:
    st.session_state['genome_file'] = None
if os.path.exists(predicted_promoter_files):
    st.session_state['train_data'] = predicted_promoter_files
    st.session_state['predicted_promoter_file'] = predicted_promoter_files
else:
    st.session_state['train_data'] = None
    st.session_state['predicted_promoter_file'] = None

if os.path.exists(uploaded_train_data):
    st.session_state['train_data'] = uploaded_train_data
else:
    st.session_state['train_data'] = None

if os.path.exists(training_args):
    st.session_state['training_parameters'] = training_args
else:
    st.session_state['training_parameters'] = None

# File Upload
# if st.session_state['factory_root']:
    # st.sidebar.markdown(f"Factory Root: {st.session_state['factory_root']}")
    # st.sidebar.markdown(f"Project Name: {project_name}")

# column_buttons = st.columns(2)

# if os.path.exists(f"{st.session_state['factory_root']}/genome_file.fasta"):
#     st.session_state['genome_file'] = f"{st.session_state['factory_root']}/genome_file.fasta"

with columns_0[1]:
    sub_columns = st.columns(2)
    with sub_columns[0]:
        st.markdown("Upload a genome file")
        genome_file = st.file_uploader("genome_file", type=["fasta"])
        # save genome_file to local cache
        if genome_file is not None:
            with open(f"{st.session_state['factory_root']}/genome_file.fasta", "wb") as f:
                f.write(genome_file.getbuffer())
            st.session_state['genome_file'] = f"{st.session_state['factory_root']}/genome_file.fasta"
        if st.session_state['genome_file']:
            # st.write(st.session_state['genome_file'])
            st.info("Genome file has been uploaded.")
    with sub_columns[1]:
        st.markdown("Upload promoter sequences in FASTA format")
        promoter_file = st.file_uploader("promoter_file", type=["fasta"])
        if promoter_file is not None:
            with open(f"{st.session_state['factory_root']}/promoter_file.fasta", "wb") as f:
                f.write(promoter_file.getbuffer())
            st.session_state['train_data'] = f"{st.session_state['factory_root']}/promoter_file.fasta"
        if st.session_state['train_data']:
            st.info("Promoter file has been uploaded.")

with columns_0[0]:
    if st.session_state['genome_file']:
        if st.button("Generate Training Data", use_container_width=True):
            with st.spinner("Running Prokka..."):
                run_prokka(st.session_state)
            st.info("Prokka has finished running.")
            with st.spinner("Extracting Upstream Sequences..."):
                extract_upstream_sequences(st.session_state)
            st.info("Upstream sequences have been extracted.")
            with st.spinner("Predicting Promoters..."):
            # threads = st.number_input("Threads", value=4)
                st.session_state['train_data'] = predict_promoters(st.session_state)
                st.session_state['predicted_promoter_file'] = st.session_state['train_data']
            st.info("Promoters have been predicted.")

st.markdown("## Training Parameters")
columns = st.columns(5)
with columns[0]:
    special_token = st.text_input("Special Token", value="<|usp|>")
with columns[1]:
    batch_size = st.number_input("Batch Size", value=32)
with columns[2]:
    learning_rate = st.text_input("Learning Rate", value="5e-5")
    learning_rate = float(learning_rate)
with columns[3]:
    warmup_steps = st.number_input("Warmup Steps", value=1000)
with columns[4]:
    logging_steps = st.number_input("Logging Steps", value=1000)

columns_2 = st.columns(5)
with columns_2[0]:
    save_steps = st.number_input("Save Steps", value=1000)
with columns_2[1]:
    eval_steps = st.number_input("Evaluation Steps", value=1000)
with columns_2[2]:
    max_steps = st.number_input("Maximum Steps", value=10000)
with columns_2[3]:
    fp16 = st.text_input("FP16 (True or False)", value="True")
    fp16 = True if fp16.lower() == "true" else False
with columns_2[4]:
    seed = st.number_input("Seed", value=42)

columns_3 = st.columns(5)
with columns_3[0]:
    base_model_name = st.selectbox("Base Model Name", ["xsmall", "small", "base"])
    base_model_name = f"jinyuan22/promogen2-{base_model_name}"
    st.session_state['base_model_name'] = base_model_name
with columns_3[1]:
    fasta_data_path = st.text_input("FASTA Data Path", value=st.session_state['train_data'])
with columns_3[2]:
    output_dir = st.text_input("Output Directory", value="finetuned_model_v0")
with columns_3[3]:
    device = st.text_input("Device", value="cuda:0")
with columns_3[4]:
    local_model_path = st.text_input("Local Model Path", value="")
if local_model_path:
    st.info("Using local model path instead of Hugging Face model.")
    st.session_state['base_model_name'] = local_model_path
    base_model_name = local_model_path

if fasta_data_path:
    if os.path.exists(fasta_data_path):
        if st.button("Save training parameters"):
            training_parameters = {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "eval_steps": eval_steps,
                "max_steps": max_steps,
                "fp16": fp16,
                "seed": seed,
                "device": device,
                "fasta_data_path": fasta_data_path,
                "special_token": special_token,
                "base_model_name": base_model_name,
                "output_dir": f"{st.session_state['factory_root']}/{output_dir}",
            }
            # os.makedirs(training_parameters['output_dir'], exist_ok=True)
            with open(f"{st.session_state['factory_root']}/training_parameters.json", "w") as f:
                json.dump(training_parameters, f)

            st.session_state['training_parameters'] = f"{st.session_state['factory_root']}/training_parameters.json"

if os.path.exists(f"{st.session_state['factory_root']}/{output_dir}"):
    # st.info("")
    st.session_state['trained_model_path'] = f"{st.session_state['factory_root']}/{output_dir}"
else:
    st.session_state['trained_model_path'] = None

eval_output_file = os.path.join(st.session_state['factory_root'], output_dir, "eval_output.json")
if os.path.exists(eval_output_file):
    st.session_state['eval_output'] = json.load(open(eval_output_file, "r"))
else:
    st.session_state['eval_output'] = None

if st.session_state['training_parameters'] is not None and os.path.exists(f"{st.session_state['factory_root']}/{output_dir}") != True:
    if st.button("Train Model"):
        with st.spinner("Training Model..."):
            args = json.load(open(st.session_state['training_parameters'], "r"))
            eval_output, train_model_path = train_model(**args)
            eval_output_file = os.path.join(st.session_state['factory_root'], output_dir, "eval_output.json")
            with open(eval_output_file, "w") as f:
                json.dump(eval_output, f)
            st.session_state['trained_model_path'] = train_model_path
            eval_output = json.load(open(eval_output_file, "r"))
            st.session_state['eval_output'] = eval_output

trained_model_path = os.path.join(st.session_state['factory_root'], output_dir)
# st.write(trained_model_path)
if not os.path.exists(trained_model_path):
    st.session_state['trained_model_path'] = None

if st.session_state['trained_model_path'] is not None:
    st.markdown(f"## Generation and Analysis using `{st.session_state['trained_model_path'].replace(st.session_state['factory_root'], '')}`")
    columns_4 = st.columns(6)
    with columns_4[0]:
        num_return_sequences = st.number_input("Number of Sequences", value=128*50)
    with columns_4[1]:
        batch_size = st.number_input("Batch Size", value=256)
    with columns_4[2]:
        max_new_tokens = st.number_input("Maximum Length", value=60)
    with columns_4[3]:
        repetition_penalty = st.number_input("Repetition Penalty", value=1.0)
    with columns_4[4]:
        temperature = st.number_input("Temperature", value=1.0)
    with columns_4[5]:
        top_p = st.number_input("Top P", value=1.0)
    generation_parameters = {
        "special_token": special_token,
        "model_path": st.session_state['trained_model_path'],
        "num_return_sequences": num_return_sequences,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "output_dir": f"{st.session_state['factory_root']}/generated_promoters",
    }
    with open(f"{st.session_state['factory_root']}/generation_parameters.json", "w") as f:
        json.dump(generation_parameters, f)
    st.session_state['generation_parameters'] = f"{st.session_state['factory_root']}/generation_parameters.json"

    if st.button("Generate Promoters"):
        generation_parameters = json.load(open(st.session_state['generation_parameters'], "r"))
        with st.spinner("Generating Promoters..."):
            output_file_name = generate_promoters(**generation_parameters)
        st.session_state['generated_promoter_file'] = output_file_name
    generated_fastas = glob(f"{st.session_state['factory_root']}/generated_promoters/*.fasta")
    if len(generated_fastas) >= 1:
        options = [os.path.basename(f) for f in generated_fastas]
        fasta_file_selected = st.selectbox("Generated Promoters", options=options)
        st.session_state['generated_promoter_file'] = os.path.join(st.session_state['factory_root'], "generated_promoters", fasta_file_selected)
    if st.button("Analyze K-mers"):
        native_promoter_file = st.session_state['train_data']
        generated_promoter_file = st.session_state['generated_promoter_file']
        output_png = analyze_kmers(native_promoter_file, generated_promoter_file)
        st.image(output_png)

generation_parameter_file = os.path.join(st.session_state['factory_root'], "generation_parameters.json")
if os.path.exists(generation_parameter_file):
    st.session_state['generation_parameters'] = generation_parameter_file
else:
    st.session_state['generation_parameters'] = None

generated_fastas = glob(f"{st.session_state['factory_root']}/generated_promoters/*.fasta")
if len(generated_fastas) >= 1:
    st.markdown("## Download Generated Promoters")
    st.markdown(f"[Download {os.path.basename(st.session_state['generated_promoter_file'])}]({st.session_state['generated_promoter_file']})")
else:
    st.session_state['generated_promoter_file'] = None

with st.expander("User Guide"):
    st.markdown("""
    ### User Guide
    1. **Upload Genome File**: Upload a `*.fasta` file containing the genome sequence.
    2. **Select Kingdom**: Choose the kingdom of the organism.
    3. **Special Token**: Provide any special token for the sequences.
    4. **Set Training Parameters**: Adjust the training parameters for the model.
    5. **Run Prokka**: Click the button to run Prokka for ORF prediction.
    6. **Extract Upstream Sequences**: Extract sequences upstream of ORFs.
    7. **Predict Promoters**: Predict promoters using Promoter Calculator.
    8. **Train Model**: Train a language model using the predicted promoters.
    9. **Generate Promoters**: Use the trained model to generate new promoter sequences.
    10. **Analyze K-mers**: Compare the k-mer distribution of native and generated promoters.
    """)


update_state_in_sidebar(st.session_state)