"""
GPT-2 Recipe Generator - Streamlit Application
Fine-tuned model for generating cooking recipes from ingredients
Fetches model from Kaggle Dataset
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🍳 AI Recipe Generator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .recipe-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .ingredient-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 1rem;
        font-size: 0.9rem;
    }
    .step-number {
        background-color: #4ECDC4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# KAGGLE DATASET CONFIGURATION
# ============================================================================

# IMPORTANT: Replace with YOUR Kaggle username and dataset name
KAGGLE_DATASET_ID = "your-username/gpt2-recipe-model"  # UPDATE THIS!

# Define paths for model files
MODEL_DIR = Path("downloaded_model")
MODEL_PATH = MODEL_DIR / "final_model"
BASE_MODEL = "gpt2"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DOWNLOAD MODEL FROM KAGGLE
# ============================================================================

@st.cache_resource
def download_and_setup_model():
    """
    Downloads model files from Kaggle if they don't exist.
    This function runs only once per session thanks to st.cache_resource.
    """
    # Check if running in Streamlit Cloud with Kaggle credentials
    if 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
        # Check if model is already downloaded
        if not MODEL_PATH.exists():
            st.info("📥 Downloading model files from Kaggle... This may take a moment.")
            
            try:
                # Setup Kaggle API credentials
                os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
                
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                # Create directory to download files
                MODEL_DIR.mkdir(exist_ok=True)
                
                # Download the dataset
                api.dataset_download_files(KAGGLE_DATASET_ID, path=MODEL_DIR, unzip=True)
                st.success("✅ Model files downloaded successfully!")
                return True
                
            except Exception as e:
                st.error(f"❌ Error downloading from Kaggle: {e}")
                st.info("Please check:\n1. KAGGLE_DATASET_ID is correct\n2. Kaggle credentials in secrets\n3. Dataset is public or you have access")
                return False
    
    # For local development
    else:
        if not MODEL_PATH.exists():
            st.warning("""
                ⚠️ **Running locally without Kaggle credentials**
                
                Please either:
                1. Add Kaggle credentials to `.streamlit/secrets.toml`:
                   ```
                   KAGGLE_USERNAME = "your_username"
                   KAGGLE_KEY = "your_api_key"
                   ```
                2. Or manually place model files in `downloaded_model/final_model/` directory
            """)
            return False
        return True
    
    return MODEL_PATH.exists()

# ============================================================================
# LOAD MODEL & TOKENIZER (CACHED)
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer from downloaded files"""
    
    # First, ensure model is downloaded
    model_ready = download_and_setup_model()
    
    if not model_ready:
        st.error("❌ Model files are not available. Cannot load the model.")
        return None, None, None
    
    with st.spinner("🔄 Loading AI model... This may take a moment..."):
        try:
            # Load tokenizer
            try:
                # Try tiktoken wrapper (if used during training)
                import tiktoken
                
                class TiktokenWrapper:
                    def __init__(self):
                        self.encoding = tiktoken.get_encoding("gpt2")
                        self.vocab_size = self.encoding.n_vocab
                        self.eos_token_id = 50256
                        self.bos_token_id = 50256
                        self.pad_token_id = 50256
                        self.eos_token = "<|endoftext|>"
                    
                    def encode(self, text, add_special_tokens=True):
                        tokens = self.encoding.encode(text)
                        if add_special_tokens:
                            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
                        return tokens
                    
                    def decode(self, tokens, skip_special_tokens=True):
                        if isinstance(tokens, torch.Tensor):
                            tokens = tokens.tolist()
                        if skip_special_tokens:
                            tokens = [t for t in tokens if t not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
                        return self.encoding.decode(tokens)
                    
                    def __len__(self):
                        return self.vocab_size
                
                tokenizer = TiktokenWrapper()
                tokenizer_type = "tiktoken"
                st.info("✅ Using Tiktoken tokenizer")
            
            except:
                # Fallback to transformers tokenizer
                st.info("📥 Loading GPT-2 tokenizer...")
                tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer_type = "transformers"
                st.info("✅ Using Transformers tokenizer")
            
            # Load base model
            st.info("📥 Loading base GPT-2 model...")
            base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
            
            # Load LoRA adapters from downloaded path
            st.info("📥 Loading fine-tuned LoRA adapters...")
            if MODEL_PATH.exists():
                model = PeftModel.from_pretrained(base_model, str(MODEL_PATH))
                st.success("✅ Fine-tuned model loaded successfully!")
            else:
                st.warning("⚠️ LoRA adapters not found. Using base GPT-2.")
                model = base_model
            
            model = model.to(DEVICE)
            model.eval()
            
            st.success(f"✅ Model ready on {DEVICE.upper()}!")
            
            return model, tokenizer, tokenizer_type
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None, None

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.tokenizer_type = None
    st.session_state.model_loaded = False

# Load model (only once)
if not st.session_state.model_loaded:
    model, tokenizer, tokenizer_type = load_model_and_tokenizer()
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.tokenizer_type = tokenizer_type
    st.session_state.model_loaded = True
else:
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    tokenizer_type = st.session_state.tokenizer_type

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

class GenerationConfig:
    MAX_LENGTH = 512
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 50

gen_config = GenerationConfig()

# ============================================================================
# RECIPE GENERATION FUNCTION
# ============================================================================

def generate_recipe(title, ingredients, temperature=0.8, top_p=0.9, top_k=50, 
                   max_length=512, num_samples=1):
    """
    Generate recipe directions from title and ingredients
    
    Args:
        title: Recipe title
        ingredients: Comma-separated ingredients
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_length: Maximum generation length
        num_samples: Number of variations to generate
    
    Returns:
        List of generated recipes
    """
    if model is None or tokenizer is None:
        return ["Model not loaded. Please check configuration."]
    
    # Format prompt
    prompt = f"Title: {title} | Ingredients: {ingredients} | Directions:"
    
    # Encode
    if tokenizer_type == "tiktoken":
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)])
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    input_ids = input_ids.to(DEVICE)
    
    # Generate
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                input_ids,
                max_length=max_length,
                min_length=len(input_ids[0]) + 30,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            
            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract directions
            if "Directions:" in generated_text:
                directions = generated_text.split("Directions:")[-1].strip()
            else:
                directions = generated_text[len(prompt):].strip()
            
            results.append(directions)
    
    return results

# ============================================================================
# PARSE DIRECTIONS INTO STEPS
# ============================================================================

def parse_directions(directions_text):
    """Parse generated directions into numbered steps"""
    steps = []
    
    # Split by "Step X:" pattern
    parts = directions_text.split("Step ")
    
    for part in parts[1:]:  # Skip first empty part
        # Remove step number and get content
        content = part.split(":", 1)
        if len(content) > 1:
            steps.append(content[1].strip())
    
    # If no steps found, split by sentences
    if not steps:
        import re
        sentences = re.split(r'[.!?]+', directions_text)
        steps = [s.strip() for s in sentences if s.strip()]
    
    return steps

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🍳 AI Recipe Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Fine-tuned GPT-2 | Create delicious recipes from ingredients</div>', unsafe_allow_html=True)
    
    # Check if model loaded
    if model is None:
        st.error("⚠️ Model failed to load. Please check your configuration.")
        st.info(f"""
        **Current Configuration:**
        - Kaggle Dataset ID: `{KAGGLE_DATASET_ID}`
        - Model Path: `{MODEL_PATH}`
        
        **Setup Instructions:**
        1. Upload your model to Kaggle as a dataset
        2. Update `KAGGLE_DATASET_ID` in the code
        3. Add Kaggle credentials to Streamlit secrets
        """)
        
        # Show setup instructions
        with st.expander("📖 Setup Guide"):
            st.markdown("""
            ### For Streamlit Cloud Deployment:
            
            1. **Upload Model to Kaggle:**
               - Go to kaggle.com/datasets
               - Click "New Dataset"
               - Upload your `final_model` folder
               - Make it public or keep private (need auth)
            
            2. **Get Kaggle API Credentials:**
               - Go to kaggle.com/settings
               - Scroll to "API" section
               - Click "Create New API Token"
               - Download `kaggle.json`
            
            3. **Add to Streamlit Secrets:**
               - In your Streamlit Cloud dashboard
               - Go to App Settings → Secrets
               - Add:
               ```toml
               KAGGLE_USERNAME = "your_username"
               KAGGLE_KEY = "your_api_key_from_kaggle.json"
               ```
            
            4. **Update Code:**
               - Set `KAGGLE_DATASET_ID = "username/dataset-name"`
            
            ### For Local Development:
            
            Create `.streamlit/secrets.toml`:
            ```toml
            KAGGLE_USERNAME = "your_username"
            KAGGLE_KEY = "your_api_key"
            ```
            """)
        return
    
    # Sidebar - Generation Settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        st.info(f"""
        🤖 **Model:** GPT-2 LoRA Fine-tuned  
        💻 **Device:** {DEVICE.upper()}  
        📦 **Source:** Kaggle Dataset
        """)
        
        st.subheader("🎛️ Advanced Options")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=gen_config.DEFAULT_TEMPERATURE,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        top_p = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=gen_config.DEFAULT_TOP_P,
            step=0.05,
            help="Probability threshold for nucleus sampling"
        )
        
        top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=gen_config.DEFAULT_TOP_K,
            step=10,
            help="Number of top tokens to consider"
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=256,
            max_value=1024,
            value=gen_config.MAX_LENGTH,
            step=64,
            help="Maximum length of generation"
        )
        
        num_variations = st.selectbox(
            "Number of Variations",
            options=[1, 2, 3],
            index=0,
            help="Generate multiple recipe variations"
        )
        
        st.divider()
        
        # Quick presets
        st.subheader("🎯 Quick Presets")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎨 Creative", use_container_width=True):
                temperature = 1.2
                top_p = 0.95
        
        with col2:
            if st.button("📖 Traditional", use_container_width=True):
                temperature = 0.5
                top_p = 0.9
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Recipe Input")
        
        # Recipe title
        recipe_title = st.text_input(
            "Recipe Title",
            placeholder="e.g., Chocolate Chip Cookies",
            help="Enter the name of your recipe"
        )
        
        # Ingredients input
        ingredients_input = st.text_area(
            "Ingredients",
            placeholder="Enter ingredients separated by commas\ne.g., flour, sugar, butter, eggs, chocolate chips",
            height=150,
            help="List all ingredients separated by commas"
        )
        
        # Quick examples
        st.subheader("💡 Try These Examples")
        
        examples = {
            "🍪 Chocolate Chip Cookies": {
                "title": "Chocolate Chip Cookies",
                "ingredients": "butter, sugar, eggs, flour, chocolate chips, vanilla extract, baking soda, salt"
            },
            "🍝 Chicken Pasta": {
                "title": "Creamy Chicken Pasta",
                "ingredients": "chicken breast, pasta, heavy cream, garlic, parmesan cheese, olive oil, salt, pepper, basil"
            },
            "🥗 Caesar Salad": {
                "title": "Classic Caesar Salad",
                "ingredients": "romaine lettuce, caesar dressing, parmesan cheese, croutons, lemon juice, black pepper"
            }
        }
        
        selected_example = st.selectbox("Choose an example:", ["Custom"] + list(examples.keys()))
        
        if selected_example != "Custom":
            recipe_title = examples[selected_example]["title"]
            ingredients_input = examples[selected_example]["ingredients"]
            st.rerun()
        
        # Generate button
        generate_btn = st.button("🚀 Generate Recipe", type="primary", use_container_width=True)
    
    with col2:
        st.header("✨ Generated Recipe")
        
        if generate_btn:
            if not recipe_title or not ingredients_input:
                st.warning("⚠️ Please enter both title and ingredients!")
            else:
                with st.spinner("🔮 Generating your recipe..."):
                    # Generate recipes
                    generated_directions = generate_recipe(
                        title=recipe_title,
                        ingredients=ingredients_input,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_length=max_length,
                        num_samples=num_variations
                    )
                    
                    # Display results
                    for idx, directions in enumerate(generated_directions):
                        if num_variations > 1:
                            st.subheader(f"📋 Variation {idx + 1}")
                        
                        # Recipe box
                        st.markdown('<div class="recipe-box">', unsafe_allow_html=True)
                        
                        # Title
                        st.markdown(f"### 🍽️ {recipe_title}")
                        
                        # Ingredients
                        st.markdown("**Ingredients:**")
                        ingredients_list = [ing.strip() for ing in ingredients_input.split(',')]
                        
                        ingredients_html = "".join([f'<span class="ingredient-tag">{ing}</span>' for ing in ingredients_list])
                        st.markdown(ingredients_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Directions
                        st.markdown("**Directions:**")
                        steps = parse_directions(directions)
                        
                        if steps:
                            for i, step in enumerate(steps, 1):
                                st.markdown(f"""
                                    <div style="display: flex; align-items: start; margin: 1rem 0;">
                                        <span class="step-number">{i}</span>
                                        <span style="flex: 1;">{step}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write(directions)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button
                        recipe_text = f"""
{recipe_title}

INGREDIENTS:
{chr(10).join(['• ' + ing for ing in ingredients_list])}

DIRECTIONS:
{chr(10).join([f'{i}. {step}' for i, step in enumerate(steps, 1)])}
                        """
                        
                        st.download_button(
                            label="📥 Download Recipe",
                            data=recipe_text,
                            file_name=f"{recipe_title.replace(' ', '_').lower()}.txt",
                            mime="text/plain",
                            key=f"download_{idx}"
                        )
                        
                        if idx < len(generated_directions) - 1:
                            st.divider()
        else:
            st.info("👈 Enter a recipe title and ingredients, then click 'Generate Recipe'")
            
            # Show sample output
            st.markdown("### 📖 Example Output")
            st.markdown("""
            <div class="recipe-box">
                <p><strong>This is where your AI-generated recipe will appear!</strong></p>
                <p>The model will create step-by-step cooking directions based on your ingredients.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🧠 Model", "GPT-2 LoRA")
    
    with col2:
        st.metric("📊 Parameters", "~124M + LoRA")
    
    with col3:
        st.metric("⚡ Status", "Ready" if model else "Error")
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🍳 <strong>AI Recipe Generator</strong> | Fine-tuned GPT-2 for Recipe Generation</p>
            <p style="font-size: 0.9rem;">Built with Streamlit • Powered by Transformers & PEFT</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
