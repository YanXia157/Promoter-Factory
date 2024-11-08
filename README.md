# Promoter-Factory

```bash
git clone https://github.com/YanXia157/Promoter-Factory.git
```

- **Software dependencies**: 
  - `prokka` for genome annotation. Install via `conda`:

```bash
conda create -n promoterfactory python=3.10
conda activate promoterfactory
conda install -c bioconda prokka
```

## Start Promoter-Factory GUI

- **Install dependencies**: 
```bash
conda activate promoterfactory
pip install biopython torch transformers[torch] datasets pandas numpy scipy seaborn matplotlib streamlit
```

- **Start GUI**:

```bash
cd promoter-factory
streamlit run app.py
```

## Dependencies of the notebook version


- **Python dependencies**:

Install the Python dependencies using the following command:

```bash
pip install biopython torch transformers[torch] datasets pandas numpy scipy seaborn matplotlib jupyter notebook
```

## Usage

For specific usage details, please refer to `notebook.ipynb`, which contains detailed examples and code demonstrations.