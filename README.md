# 🍳 AI Recipe Generator with Fine-tuned GPT-2

A beautiful web application that generates detailed cooking directions from recipe titles and ingredients using a fine-tuned GPT-2 model with LoRA.

## ✨ Features

- 🤖 Fine-tuned GPT-2 model trained on 157,132 recipes
- ⚡ Fast generation (2-5 seconds per recipe)
- 🎨 Modern, responsive web interface
- 🔧 Customizable generation parameters
- 📥 Download recipes as text files
- 🎲 Generate multiple recipe variations
- 💾 Memory-efficient LoRA implementation

## 📁 Project Structure
```
ai-recipe-generator/
│
├── app.py                 # Streamlit web application
├── style.css             # Custom styling for the UI
├── finetune.ipynb        # Complete training pipeline
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended for training (14.7 GB VRAM used)
- 4GB+ disk space

### Installation
```bash
# Clone the repository
git clone https://github.com/NadeemAhmad3/GPT2_Finetunning.git
cd GPT2_Finetunning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📦 Requirements
```
streamlit>=1.38.0
torch>=2.5.0
torchvision>=0.21.0
transformers>=4.35.0
peft>=0.7.1
tiktoken>=0.5.2
numpy>=1.24.3
pandas>=2.1.4
kaggle>=1.6.0
```

## 🎓 Model Training

The model was trained on the Recipe NLG Dataset from Kaggle using the `finetune.ipynb` notebook.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | GPT-2 (124M parameters) |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) |
| Training Samples | 157,132 recipes |
| Validation Samples | 19,641 recipes |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Training Epochs | 1 |
| Batch Size | 1 (effective: 16) |
| Learning Rate | 2e-4 |
| Max Sequence Length | 512 tokens |
| Training Time | 5.29 hours |
| GPU Memory Used | 8.9 GB / 19 GB |

### Training Results
```
📊 Final Metrics:
├── Training Steps: 9,820
├── Validation Loss: 0.8479
├── Perplexity: 2.33
├── Trainable Parameters: 294,912 (0.24%)
└── Model Size: 8.1 MB
```

### Memory Optimization

- **LoRA adapters**: Only 0.24% of parameters trained
- **FP16 Mixed Precision**: 50% memory reduction
- **Gradient Checkpointing**: Reduces activation memory
- **Memory-mapped data**: Efficient data loading

## 🖥️ Using the Application

1. **Enter Recipe Details**
   - Recipe title (e.g., "Chocolate Chip Cookies")
   - Ingredients (comma-separated list)

2. **Adjust Settings** (Optional)
   - Temperature (0.1-2.0): Controls creativity
   - Top P (0.1-1.0): Nucleus sampling
   - Top K (10-100): Limits vocabulary
   - Max Length (256-1024): Output length
   - Variations (1-3): Number of recipes

3. **Generate**
   - Click "🚀 Generate Recipe"
   - View generated directions
   - Download as text file

## 📊 Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.8 | Higher = more creative |
| Top P | 0.9 | Nucleus sampling threshold |
| Top K | 50 | Top-k sampling limit |
| Max Length | 512 | Maximum tokens to generate |

## 🎯 Example Output

**Input:**
```
Title: Chocolate Chip Cookies
Ingredients: butter, sugar, eggs, flour, chocolate chips, vanilla extract
```

**Generated:**
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

## 🔧 Technical Details

### Architecture
```
GPT-2 Base Model (124M params)
    ↓
LoRA Adapters (294K params)
    ↓
Fine-tuned on 157K recipes
    ↓
Recipe Generation
```

### Tokenization

- **Primary**: Tiktoken (offline-compatible)
- **Fallback**: Transformers GPT2Tokenizer
- **Vocab Size**: 50,257 tokens

## 🐛 Troubleshooting

**Model not loading:**
```bash
# Check model files in downloaded_model/
# Should contain: adapter_config.json, adapter_model.bin
```

**Out of memory:**
```python
# Reduce max_length in generation settings
max_length = 256  # Instead of 512
```

**Slow generation:**
```bash
# Ensure GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Future Improvements

- [ ] Add recipe categories
- [ ] Nutritional information generation
- [ ] Multi-cuisine support
- [ ] Recipe image generation
- [ ] API endpoint
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Recipe NLG Dataset from Kaggle
- OpenAI GPT-2
- Hugging Face Transformers
- Microsoft PEFT (LoRA)
- Streamlit

## 📧 Contact

- GitHub: [@yourusername](https://github.com/NadeemAhmad3)
- Email: nadeemahmad2703@gmail.com

---

