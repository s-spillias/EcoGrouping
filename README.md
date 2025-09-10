# EcoGrouping: Benchmarking Large Language Models for Ecological Functional Group Classification

This repository contains the code and analysis pipeline for benchmarking Large Language Models (LLMs) on ecological functional group classification tasks. The project evaluates how well different AI models can classify marine taxa into functional groups and create meaningful ecological groupings.

## Overview

The EcoGrouping project implements two main experimental approaches:

1. **Emergent Classification** (`Group.py` + `Group-Analysis.py`): Evaluates LLMs' capacity to create functional groups from scratch given a list of taxa

2. **Predefined Classification** (`Classify.py` + `Classify-Analysis.py`): Tests LLMs' ability to classify individual taxa into predefined functional groups


## Project Structure

```
EcoGrouping/
├── README.md                    # This file
├── Group.py                     # Emergent grouping experiments  
├── Group-Analysis.py            # Analysis of grouping results
├── Classify.py                  # Predefined classification experiments
├── Classify-Analysis.py         # Analysis of classification results
├── Datasets/                    # Reference ecological datasets
│   └── Groupings/              # Taxon-to-group mappings (JSON)
├── Artifacts/                   # Generated results and outputs
│   ├── Classify/               # Classification experiment results
│   └── Group/                  # Grouping experiment results
├── Figures/                    # Generated plots and visualizations
└── scripts/                    # Helper functions and utilities
```

## Core Scripts

### 1. Group.py - Emergent Grouping

**Purpose**: Tests LLMs' ability to create functional groups from scratch.

**Key Features**:
- Generates functional group proposals using structured prompts
- Supports multiple models and replicates
- Implements retry logic for robust JSON extraction
- Saves raw responses for detailed analysis

**Usage**:
```bash
python Group.py [model_name]
```

**Key Parameters**:
- `replicate`: Number of replicates per dataset-model combination (default: 5)
- `models`: List of LLM models to evaluate

### 2. Group-Analysis.py - Grouping Analysis

**Purpose**: Analyzes emergent grouping results and compares with reference classifications.

**Key Features**:
- Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
- Implements robust ARI calculation with coverage weighting
- Performs fuzzy matching for taxonomic name variations
- Generates comparative visualizations

**Key Functions**:
- `compute_adjusted_rand_score()`: Calculates ARI between generated and reference groups
- `robust_adjusted_rand_score()`: Implements coverage-weighted robust ARI
- `pairwise_cluster_f1()`: Computes pairwise clustering F1 scores
- `plot_ami_vs_ari()`: Creates scatter plots comparing AMI and ARI

### 3. Classify.py - Supervised Classification

**Purpose**: Tests LLMs' ability to classify individual taxa into predefined functional groups.

**Key Features**:
- Processes taxa in configurable batch sizes (1, 2, 4, 8, 16, 32, 64-128, 256)
- Supports multiple iterations per batch size for statistical robustness
- Implements health checks to detect model breakdown at larger batch sizes
- Saves results in structured JSON format with metadata

**Usage**:
```bash
python Classify.py [model_name]
```

**Key Parameters**:
- `iterations_per_size`: Number of replicates per batch size (default: 5)
- `chunk_sizes`: List of batch sizes to test
- `min_completion_rate`: Minimum success rate before flagging breakdown (default: 0.85)
- `max_failed_chunks`: Maximum failed chunks before stopping (default: 1)

### 4. Classify-Analysis.py - Classification Analysis

**Purpose**: Analyzes results from supervised classification experiments.

**Key Features**:
- Computes classification metrics (precision, recall, F1-score, ARI)
- Generates visualizations of model performance
- Performs statistical modeling of batch size effects

**Key Functions**:
- `run_agreement_analysis()`: Main analysis pipeline
- `calculate_metrics_by_chunk_size()`: Aggregates metrics by batch size
- `plot_supervised_bar()`: Creates grouped bar charts of performance
- `plot_dumbbells_by_dataset()`: Compares supervised vs emergent approaches

## Datasets

The project uses marine ecosystem datasets with predefined functional group classifications:

- **Allain_WarmPool.json**: Warm pool ecosystem (119 taxa)
- **Chagaris_WestFloridaShelf.json**: West Florida Shelf (100 taxa)
- **Dahood_WestAntarctic.json**: West Antarctic ecosystem (43 taxa)
- **Geers_GulfofMexico.json**: Gulf of Mexico (100 taxa)
- **Mcmullen_Penguins.json**: Penguin ecosystem (17 taxa)
- **Montero_CanaryIslands.json**: Canary Islands (100 taxa)

Each dataset contains taxon-to-functional-group mappings used as ground truth for evaluation.

## Supported Models

The framework uses various LLM providers:

- **Claude**: `claude-sonnet-4-20250514`
- **Gemini**: `gemini-2.5-flash`
- **Ollama Models**: Various open-source models via Ollama

## Key Metrics

### Classification Metrics
- **Macro F1-Score**: Average F1 across functional groups
- **Robust Adjusted Rand Index (ARI†)**: Coverage-weighted clustering agreement

### Grouping Metrics
- **Adjusted Rand Index (ARI)**: Measures clustering similarity
- **Adjusted Mutual Information (AMI)**: Information-theoretic clustering measure
- **Pairwise F1**: Precision/recall on pairwise co-clustering decisions

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/########/EcoGrouping.git
cd EcoGrouping
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
```

Edit the `.env` file with your API credentials:

```bash
# Required for Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Required for OpenRouter models (GPT, etc.)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Required for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Required for Ollama (local models)
OLLAMA_PATH=/path/to/ollama/executable
OLLAMA_MODELS_PATH=/path/to/ollama/models/directory
```

**API Key Setup:**
- **Anthropic API**: Get your key from [console.anthropic.com](https://console.anthropic.com)
- **OpenRouter API**: Get your key from [openrouter.ai](https://openrouter.ai/keys)
- **Google API**: Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 4. Set Up Ollama (Optional - for Local Models)

Ollama allows you to run open-source models locally. This is optional but recommended for testing with models like DeepSeek, Qwen, and others.

#### Install Ollama

**On macOS:**
```bash
brew install ollama
```

**On Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**On Windows:**
Download from [ollama.ai](https://ollama.ai/download)

#### Download Models

After installing Ollama, download the models used in this project:

```bash
# Core models used in experiments
ollama pull deepseek-r1:70b
ollama pull deepseek-r1:14b
ollama pull gpt-oss:latest
ollama pull gpt-oss:120b
ollama pull qwen3:235b-a22b-instruct-2507-q4_K_M
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama pull gemma3n:latest
```

**Note**: These are large models (several GB each). Download only the ones you plan to use.

#### Configure Ollama Paths

Find your Ollama installation paths:
```bash
# Find Ollama executable
which ollama

# Find models directory (usually ~/.ollama/models)
ollama list
```

Update your `.env` file with these paths:
```bash
OLLAMA_PATH=/usr/local/bin/ollama  # or your ollama path
OLLAMA_MODELS_PATH=/Users/yourusername/.ollama/models  # or your models path
```

#### Start Ollama Service

```bash
# Start Ollama server (runs in background)
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

## Running Experiments

### Full Pipeline
```bash
# Run supervised classification
python Classify.py

# Analyze classification results
python Classify-Analysis.py

# Run emergent grouping
python Group.py

# Analyze grouping results
python Group-Analysis.py
```

### Single Model Testing
```bash
# Test specific model
python Classify.py claude-sonnet-4-20250514
python Group.py gemini-2.5-flash
```

## Output Files

### Classification Results
- `Artifacts/Classify/Results.json`: Detailed classification results
- `Artifacts/Classify/Results-Summary.json`: Aggregated metrics
- `Artifacts/Classify/Results-Qualitative.json`: Per-taxon analysis

### Grouping Results
- `Artifacts/Group/Output-Raw.json`: Raw grouping proposals
- `Artifacts/Group/Summary.json`: Comparative analysis results
- `Artifacts/Group/Groups.json`: Statistical summaries

### Visualizations
- `Figures/Figure_Supervised.png`: Supervised classification performance
- `Figures/Figure_Group_Accuracy.png`: Per-dataset accuracy heatmaps
- `Figures/Figure_AMI_vs_ARI.png`: AMI vs ARI scatter plots
- `Figures/Figure_Dumbbells_ByDataset.png`: Supervised vs emergent comparison

## Research Applications

This framework enables research into:

- **Model Capabilities**: How well do LLMs understand ecological relationships?
- **Scaling Effects**: How does performance change with batch size?
- **Domain Transfer**: Can models generalize across different ecosystems?


