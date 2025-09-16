# LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2509.09614-b31b1b.svg)](https://arxiv.org/abs/2509.09614)

**LoCoBench** is a comprehensive benchmark specifically designed to evaluate long-context Large Language Models (LLMs) in complex software development scenarios. It provides 8,000 evaluation scenarios across 10 programming languages with context lengths spanning 10K to 1M tokens.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/SalesforceAIResearch/LoCoBench.git
cd LoCoBench

# Install dependencies
pip install -r requirements.txt

# Install LoCoBench package
pip install -e .
```

### Download Evaluation Data

Download the complete evaluation dataset (data.zip):

```bash
# Download data.zip from Google Drive
# Visit: https://drive.google.com/file/d/1pK1M1sRrVZUDMKYcwh49CdXug0UzStvl/view?usp=sharing
# Or use gdown (install with: pip install gdown)
gdown https://drive.google.com/uc?id=1pK1M1sRrVZUDMKYcwh49CdXug0UzStvl

# Extract the data
unzip data.zip

# This will create the data/ directory with all evaluation scenarios
```

### Environment Setup

1. **Configure API Keys**

Create an `api.sh` file (gitignored) with your LLM API credentials:

```bash
# Copy the template
cp api.sh.template api.sh

# Edit api.sh with your API keys
export OPENAI_API_KEY="your_openai_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export GOOGLE_API_KEY="your_google_key_here"

# Source the file
source api.sh
```


## 📊 Running Evaluations

### Option 1: Quick Evaluation (Recommended)

Run evaluation on all LoCoBench scenarios:

```bash
# Evaluate a single model on all scenarios
locobench evaluate --model gpt-4o --config-path config.yaml

# Evaluate specific task categories
locobench evaluate --model claude-sonnet-4 --task-category architectural_understanding --difficulty hard

# Evaluate multiple models in parallel
locobench evaluate --model gpt-4o,claude-sonnet-4,gemini-2.5-pro --config-path config.yaml
```

### Option 2: Custom Evaluation

```bash
# Evaluate on specific programming languages
locobench evaluate --model gpt-4o --languages python,java,cpp

# Evaluate specific domains
locobench evaluate --model gemini-2.5-pro --domains web_applications,ml_systems
```

### Evaluation Results

Results are saved in `evaluation_results/` directory:

```
evaluation_results/
├── gpt4o_evaluation_results.json          # Detailed results
└── gpt4o_evaluation_results_summary.md    # Human-readable summary
```

## 📈 Understanding Results

### LoCoBench Score (LCBS)

The unified score (0-5 scale) combines 17 metrics across 4 dimensions:

- **Software Engineering Excellence** (40%): ACS, DTA, CFRD, STS, RS, CS, IS, SES
- **Functional Correctness** (30%): Compilation, Unit Tests, Integration Tests, IDC  
- **Code Quality Assessment** (20%): Security Analysis, Code Issues, Style Adherence
- **Long-Context Utilization** (10%): ICU, MMR

### Key Metrics Explained

- **ACS (Architectural Coherence Score)**: System-level design consistency
- **DTA (Dependency Traversal Accuracy)**: Cross-file reasoning ability
- **CFRD (Cross-File Reasoning Depth)**: Multi-file understanding
- **ICU (Information Coverage Utilization)**: Effective use of long context
- **MMR (Multi-Session Memory Retention)**: Context persistence across sessions


## 📚 Documentation

- **[Generation Guide](LoCoBench_generation.md)**: How to generate custom scenarios (Phases 1-4)
- **[Contributing](CONTRIBUTING.md)**: How to contribute to LoCoBench

## 📄 Citation

```bibtex
@article{locobench2024,
  title={LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering},
  author={Qiu, Jielin and Liu, Zuxin and Liu, Zhiwei and Murthy, Rithesh and Zhang, Jianguo and Chen, Haolin and Wang, Shiyu and Zhu, Ming and Yang, Liangwei and Tan, Juntao and Cen, Zhepeng and Qian, Cheng and Heinecke, Shelby and Yao, Weiran and Savarese, Silvio and Xiong, Caiming and Wang, Huan},
  journal={arXiv preprint arXiv:2509.09614},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Salesforce AI Research for supporting this research
- The open-source community for various tools and libraries used in this project

