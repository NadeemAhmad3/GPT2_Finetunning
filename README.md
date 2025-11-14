<div align="center">

# 🍳 AI Recipe Generator

### Transform Ingredients into Culinary Instructions with AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/)

[Live Demo](https://recipify-gpt2finetunning.streamlit.app/) • [Documentation](#-table-of-contents) • [Report Bug](https://github.com/NadeemAhmad3/GPT2_Finetunning/issues) • [Request Feature](https://github.com/NadeemAhmad3/GPT2_Finetunning/issues)

---

### 📸 Application Preview

<!-- Add your screenshots here -->
<img width="1907" height="958" alt="image" src="https://github.com/user-attachments/assets/9b7633ad-30a2-4108-9f56-173fcf856cb9" />
*Modern, intuitive interface for recipe generation*

<img width="1919" height="888" alt="image" src="https://github.com/user-attachments/assets/16e8ae47-1dac-40ec-978d-8b0820ea74a9" />
*AI-generated cooking directions from ingredients*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Model Training](#-model-training)
- [Usage Guide](#-usage-guide)
- [Generation Parameters](#-generation-parameters)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

The **AI Recipe Generator** is a state-of-the-art web application that leverages a fine-tuned GPT-2 language model to generate detailed, step-by-step cooking directions from recipe titles and ingredient lists. Built with modern machine learning techniques and optimized for performance, this project demonstrates the practical application of Large Language Models (LLMs) in the culinary domain.

### What Makes This Special?

- **Fine-tuned on 157,132+ recipes** from the Recipe NLG Dataset
- **Memory-efficient LoRA** (Low-Rank Adaptation) implementation
- **Fast generation** - Get complete recipes in 2-5 seconds
- **Production-ready** web interface built with Streamlit
- **Highly customizable** generation parameters for varied outputs

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered Generation
- Fine-tuned GPT-2 (124M parameters)
- LoRA adapters for efficient training
- Context-aware recipe generation
- Multiple variation support

</td>
<td width="50%">

### 🎨 User Experience
- Modern, responsive UI design
- Real-time recipe generation
- Customizable parameters
- Download as text files

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Performance
- 2-5 second generation time
- GPU-accelerated inference
- Memory-optimized architecture
- Efficient tokenization

</td>
<td width="50%">

### 🔧 Developer-Friendly
- Well-documented codebase
- Modular architecture
- Easy deployment
- Comprehensive training notebook

</td>
</tr>
</table>

---

## 🛠 Tech Stack

### Core Technologies

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **ML Framework** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) |
| **Transformers** | ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) |
| **Web Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| **Fine-tuning** | ![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange?style=for-the-badge) |

</div>

### Dependencies

```yaml
Machine Learning:
  - torch >= 2.5.0          # Deep learning framework
  - torchvision >= 0.21.0   # Computer vision utilities
  - transformers >= 4.35.0  # Hugging Face transformers
  - peft >= 0.7.1           # Parameter-Efficient Fine-Tuning
  
Tokenization:
  - tiktoken >= 0.5.2       # Fast tokenization
  
Data Processing:
  - numpy >= 1.24.3         # Numerical computing
  - pandas >= 2.1.4         # Data manipulation
  
Web Interface:
  - streamlit >= 1.38.0     # Web app framework
  
Dataset:
  - kaggle >= 1.6.0         # Dataset download
```

---

## 🎥 Demo

### 🌐 Live Application

**Try it now:** [AI Recipe Generator Live Demo](#)

### 📹 Video Walkthrough

<!-- Add video demo link or GIF -->
![Demo GIF](screenshots/demo.gif)

### 💡 Example Generation

<details>
<summary><b>Click to see example</b></summary>

**Input:**
```
Title: Chocolate Chip Cookies
Ingredients: butter, sugar, eggs, flour, chocolate chips, vanilla extract
```

**Generated Output:**
```
Step 1: Preheat oven to 350°F and line baking sheet with parchment paper.
Step 2: Cream together butter and sugar until light and fluffy.
Step 3: Beat in eggs one at a time, then add vanilla extract.
Step 4: Gradually mix in flour until well combined.
Step 5: Fold in chocolate chips evenly throughout the dough.
Step 6: Drop rounded tablespoons onto prepared baking sheet.
Step 7: Bake for 10-12 minutes until edges are golden brown.
Step 8: Cool for 5 minutes before transferring to wire rack.
```

</details>

---

## 🏗 Architecture

### System Architecture

```mermaid
graph LR
    A[User Input] --> B[Streamlit Interface]
    B --> C[Tokenization]
    C --> D[GPT-2 Base Model]
    D --> E[LoRA Adapters]
    E --> F[Generation]
    F --> G[Post-processing]
    G --> H[Recipe Output]
```

### Model Architecture

```
┌─────────────────────────────────────┐
│     GPT-2 Base Model (124M)        │
│     ↓                               │
│     LoRA Adapters (294K)           │
│     • Rank: 8                      │
│     • Alpha: 16                    │
│     • Trainable: 0.24%             │
│     ↓                               │
│     Fine-tuned on 157K Recipes     │
│     ↓                               │
│     Recipe Generation Engine       │
└─────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
  ```bash
  python --version
  ```

- **pip** (Python package installer)
  ```bash
  pip --version
  ```

- **Git**
  ```bash
  git --version
  ```

- **(Optional but recommended) CUDA-capable GPU**
  - For training: 14.7 GB VRAM minimum
  - For inference: 4+ GB VRAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NadeemAhmad3/GPT2_Finetunning.git
   cd GPT2_Finetunning
   ```

2. **Create a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
   ```

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to: `http://localhost:8501`
   - The app will automatically open in your default browser

3. **Using a different port (optional)**
   ```bash
   streamlit run app.py --server.port 8080
   ```

---

## 🎓 Model Training

### Dataset

The model was trained on the **Recipe NLG Dataset** from Kaggle:
- **Total Recipes:** 176,773
- **Training Set:** 157,132 recipes (88.9%)
- **Validation Set:** 19,641 recipes (11.1%)
- **Source:** [Recipe NLG Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)

### Training Configuration

<table>
<tr>
<td width="50%">

#### Model Parameters
| Parameter | Value |
|-----------|-------|
| Base Model | GPT-2 (124M) |
| Fine-tuning | LoRA |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Target Modules | `c_attn` |
| Trainable Params | 294,912 (0.24%) |

</td>
<td width="50%">

#### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 1 |
| Gradient Accum. | 16 |
| Learning Rate | 2e-4 |
| Max Seq Length | 512 |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |

</td>
</tr>
</table>

### Training Process

Run the complete training pipeline using the provided notebook:

```bash
jupyter notebook finetune.ipynb
```

**Key Steps:**
1. Dataset download and preprocessing
2. Model initialization with LoRA
3. Training loop with gradient accumulation
4. Validation and metric tracking
5. Model saving and optimization

### Training Results

```
📊 Final Training Metrics
├── Total Steps: 9,820
├── Training Time: 5.29 hours
├── Final Training Loss: 0.8479
├── Validation Loss: 0.8479
├── Perplexity: 2.33
├── GPU Memory Used: 8.9 GB / 19 GB
└── Model Size: 8.1 MB (LoRA adapters only)
```

### Memory Optimization Techniques

- **LoRA Adapters**: Train only 0.24% of parameters
- **FP16 Mixed Precision**: 50% memory reduction
- **Gradient Checkpointing**: Reduces activation memory
- **Memory-mapped Loading**: Efficient data streaming
- **Gradient Accumulation**: Simulate larger batch sizes

---

## 📖 Usage Guide

### Basic Usage

1. **Enter Recipe Information**
   - **Title**: Name of the dish (e.g., "Chocolate Chip Cookies")
   - **Ingredients**: Comma-separated list (e.g., "butter, sugar, eggs, flour")

2. **Adjust Generation Settings** (Optional)
   - Customize parameters in the sidebar
   - See [Generation Parameters](#-generation-parameters) for details

3. **Generate Recipe**
   - Click the "🚀 Generate Recipe" button
   - Wait 2-5 seconds for generation
   - Review the generated cooking directions

4. **Download or Copy**
   - Use the download button to save as `.txt`
   - Copy directly from the interface

### Advanced Features

#### Generate Multiple Variations
```python
# Set variations to 2 or 3 in the sidebar
# Each variation will have slightly different instructions
```

#### Customize Creativity
```python
# Adjust temperature:
# - Low (0.5-0.7): More deterministic, safer recipes
# - Medium (0.7-0.9): Balanced creativity
# - High (0.9-1.5): More creative, experimental recipes
```

---

## 🎛 Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Temperature** | 0.1 - 2.0 | 0.8 | Controls randomness. Higher = more creative |
| **Top P** | 0.1 - 1.0 | 0.9 | Nucleus sampling threshold. Lower = more focused |
| **Top K** | 10 - 100 | 50 | Limits vocabulary selection. Lower = more conservative |
| **Max Length** | 256 - 1024 | 512 | Maximum tokens to generate |
| **Variations** | 1 - 3 | 1 | Number of recipe variations to generate |

### Parameter Guidelines

<details>
<summary><b>When to adjust Temperature</b></summary>

- **0.5-0.7**: Standard recipes, traditional dishes
- **0.7-0.9**: Balanced creativity, everyday cooking
- **0.9-1.5**: Experimental fusion, creative variations
</details>

<details>
<summary><b>When to adjust Top P</b></summary>

- **0.8-0.9**: Most common usage
- **0.9-0.95**: More diverse vocabulary
- **0.95-1.0**: Maximum diversity (may reduce coherence)
</details>

<details>
<summary><b>When to adjust Max Length</b></summary>

- **256-384**: Simple recipes, quick dishes
- **384-512**: Standard recipes (recommended)
- **512-1024**: Complex multi-step recipes
</details>

---

## 📁 Project Structure

```
ai-recipe-generator/
│
├── 📄 app.py                      # Main Streamlit application
├── 🎨 style.css                   # Custom CSS styling
├── 📓 finetune.ipynb              # Complete training pipeline
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                   # Documentation (this file)
│
├── 📂 downloaded_model/           # Fine-tuned model directory
│   ├── adapter_config.json        # LoRA configuration
│   └── adapter_model.bin          # LoRA weights
│
├── 📂 screenshots/                # Application screenshots
│   ├── app-interface.png
│   ├── generated-recipe.png
│   └── demo.gif
│
└── 📂 data/                       # Dataset directory (gitignored)
    ├── train.csv
    └── val.csv
```

---

## 📊 Performance Metrics

### Generation Speed

| Metric | Value |
|--------|-------|
| Average Generation Time | 2-5 seconds |
| Tokens per Second | ~100-150 |
| GPU Utilization | 70-85% |
| CPU Fallback Speed | 10-20 seconds |

### Model Quality

| Metric | Score |
|--------|-------|
| Validation Loss | 0.8479 |
| Perplexity | 2.33 |
| BLEU Score | N/A (generative task) |
| Human Evaluation | Subjective |

### Resource Usage

| Resource | Training | Inference |
|----------|----------|-----------|
| GPU Memory | 8.9 GB | 2-4 GB |
| CPU Memory | 16 GB | 4-8 GB |
| Disk Space | 50 GB | 5 GB |

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>Model not loading / File not found error</b></summary>

**Problem:** Missing model files in `downloaded_model/`

**Solution:**
```bash
# Check if model files exist
ls downloaded_model/
# Should contain: adapter_config.json, adapter_model.bin

# If missing, retrain the model using finetune.ipynb
# or download pre-trained weights
```
</details>

<details>
<summary><b>CUDA out of memory error</b></summary>

**Problem:** GPU memory insufficient

**Solution:**
```python
# In app.py, reduce max_length
max_length = 256  # Instead of 512

# Or force CPU usage
device = "cpu"
```
</details>

<details>
<summary><b>Slow generation speed</b></summary>

**Problem:** Running on CPU or GPU not detected

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
</details>

<details>
<summary><b>Import errors</b></summary>

**Problem:** Missing dependencies

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Or install individually
pip install streamlit torch transformers peft
```
</details>

<details>
<summary><b>Poor quality generations</b></summary>

**Problem:** Nonsensical or incomplete recipes

**Solution:**
- Increase `max_length` to 512 or higher
- Adjust `temperature` (try 0.7-0.9)
- Provide more detailed ingredient lists
- Ensure model is properly loaded
</details>

### Getting Help

If you encounter issues not listed here:

1. Check [existing issues](https://github.com/NadeemAhmad3/GPT2_Finetunning/issues)
2. Search [Stack Overflow](https://stackoverflow.com/questions/tagged/gpt-2)
3. Create a [new issue](https://github.com/NadeemAhmad3/GPT2_Finetunning/issues/new) with:
   - Error message
   - Python version
   - System specifications
   - Steps to reproduce

---

## 🗺 Roadmap

### Current Version (v1.0)
- ✅ Fine-tuned GPT-2 model
- ✅ Streamlit web interface
- ✅ Multiple recipe variations
- ✅ Customizable parameters

### Planned Features

#### v1.1 (Short-term)
- [ ] Recipe categorization (desserts, main courses, etc.)
- [ ] Ingredient substitution suggestions
- [ ] Cooking time estimation
- [ ] Difficulty level classification

#### v1.2 (Mid-term)
- [ ] Nutritional information generation
- [ ] Multi-cuisine support (Italian, Chinese, Indian, etc.)
- [ ] Recipe scaling (servings adjustment)
- [ ] User authentication and recipe saving

#### v2.0 (Long-term)
- [ ] Recipe image generation (Stable Diffusion integration)
- [ ] REST API endpoint for integration
- [ ] Multi-language support
- [ ] Mobile application (React Native)
- [ ] Social features (sharing, ratings, comments)

### Community Requests
Vote for features by reacting to [this issue](https://github.com/NadeemAhmad3/GPT2_Finetunning/issues)!

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how you can contribute:

### Ways to Contribute

- 🐛 **Report Bugs**: Open an issue describing the bug
- 💡 **Suggest Features**: Share your ideas in issues
- 📝 **Improve Documentation**: Fix typos or add examples
- 💻 **Submit Code**: Create pull requests with improvements

### Contribution Guidelines

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/GPT2_Finetunning.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Describe your changes clearly

### Code Standards

- Follow PEP 8 style guide for Python
- Write clear commit messages
- Add docstrings to functions
- Include tests for new features (when applicable)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Nadeem Ahmad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📞 Contact

<div align="center">

### Nadeem Ahmad

[![GitHub](https://img.shields.io/badge/GitHub-NadeemAhmad3-181717?style=for-the-badge&logo=github)](https://github.com/NadeemAhmad3)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/your-linkedin-profile)
[![Email](https://img.shields.io/badge/Email-nadeemahmad2703@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nadeemahmad2703@gmail.com)

**Project Link:** [https://github.com/NadeemAhmad3/GPT2_Finetunning](https://github.com/NadeemAhmad3/GPT2_Finetunning)

</div>

---

## 🙏 Acknowledgments

Special thanks to the following projects and communities:

- **[Recipe NLG Dataset](https://www.kaggle.com/datasets/paultimothymooney/recipenlg)** - Training data source
- **[OpenAI](https://openai.com/)** - GPT-2 base model
- **[Hugging Face](https://huggingface.co/)** - Transformers library
- **[Microsoft Research](https://github.com/microsoft/LoRA)** - LoRA implementation
- **[Streamlit](https://streamlit.io/)** - Web framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

### Inspiration & Resources

- [How to Fine-tune GPT-2](https://huggingface.co/blog/how-to-generate)
- [LoRA: Low-Rank Adaptation Paper](https://arxiv.org/abs/2106.09685)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by [Nadeem Ahmad](https://github.com/NadeemAhmad3)

</div>
