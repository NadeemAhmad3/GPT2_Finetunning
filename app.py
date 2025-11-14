"""
GPT-2 Recipe Generator - Modern Web Interface
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
    initial_sidebar_state="collapsed"
)

# ============================================================================
# LOAD CUSTOM CSS
# ============================================================================

if os.path.exists('style.css'):
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ============================================================================
# KAGGLE DATASET CONFIGURATION
# ============================================================================

KAGGLE_DATASET_ID = "nadeemahmad003/gpt2-finetunning"
MODEL_DIR = Path("downloaded_model")
BASE_MODEL = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DOWNLOAD MODEL FROM KAGGLE
# ============================================================================

@st.cache_resource
def download_and_setup_model():
    """Downloads model files from Kaggle if they don't exist."""
    if 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
        adapter_config = MODEL_DIR / "adapter_config.json"
        
        if not adapter_config.exists():
            try:
                os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
                
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                MODEL_DIR.mkdir(exist_ok=True)
                api.dataset_download_files(KAGGLE_DATASET_ID, path=MODEL_DIR, unzip=True)
                return True
                
            except Exception as e:
                st.error(f"❌ Error downloading from Kaggle: {e}")
                return False
    else:
        adapter_config = MODEL_DIR / "adapter_config.json"
        if not adapter_config.exists():
            return False
        return True
    
    return (MODEL_DIR / "adapter_config.json").exists()

# ============================================================================
# LOAD MODEL & TOKENIZER
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer from downloaded files"""
    
    model_ready = download_and_setup_model()
    
    if not model_ready:
        st.error("❌ Model files are not available. Cannot load the model.")
        return None, None, None
    
    try:
        try:
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
        
        except:
            tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer_type = "transformers"
        
        base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
        
        if (MODEL_DIR / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
        else:
            st.warning("⚠️ LoRA adapters not found. Using base GPT-2.")
            model = base_model
        
        model = model.to(DEVICE)
        model.eval()
        
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
# RECIPE GENERATION FUNCTION
# ============================================================================

def generate_recipe(title, ingredients, temperature=0.8, top_p=0.9, top_k=50, 
                   max_length=512, num_samples=1):
    """Generate recipe directions from title and ingredients"""
    if model is None or tokenizer is None:
        return ["Model not loaded. Please check configuration."]
    
    prompt = f"Title: {title} | Ingredients: {ingredients} | Directions:"
    
    if tokenizer_type == "tiktoken":
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)])
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    input_ids = input_ids.to(DEVICE)
    
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
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
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
    parts = directions_text.split("Step ")
    
    for part in parts[1:]:
        content = part.split(":", 1)
        if len(content) > 1:
            steps.append(content[1].strip())
    
    if not steps:
        import re
        sentences = re.split(r'[.!?]+', directions_text)
        steps = [s.strip() for s in sentences if s.strip()]
    
    return steps

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Navigation Bar
    st.markdown("""
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-logo">🍳 AI Recipe Generator</div>
            <div class="nav-links">
                <a href="#home">Home</a>
                <a href="#generate">Generate</a>
                <a href="#about">About</a>
            </div>
        </div>
    </nav>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <section class="hero" id="home">
        <div class="hero-content">
            <h1 class="hero-title">Create Delicious Recipes<br/>with AI Magic</h1>
            <p class="hero-subtitle">Just enter a recipe title and ingredients, and let our AI powered by fine-tuned GPT-2 generate detailed cooking directions for you.</p>
            <div class="hero-stats">
                <div class="stat-item">
                    <div class="stat-number">GPT-2</div>
                    <div class="stat-label">Fine-tuned Model</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">∞</div>
                    <div class="stat-label">Recipe Ideas</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">&lt;5s</div>
                    <div class="stat-label">Generation Time</div>
                </div>
            </div>
        </div>
        <div class="hero-image">
            <div class="food-grid">
                <div class="food-item">🍰</div>
                <div class="food-item">🍕</div>
                <div class="food-item">🍔</div>
                <div class="food-item">🍜</div>
                <div class="food-item">🥗</div>
                <div class="food-item">🍱</div>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)
    
    # Check if model loaded
    if model is None:
        st.markdown("""
        <section class="error-section">
            <div class="error-container">
                <h2>⚠️ Model Not Available</h2>
                <p>The AI model could not be loaded. Please check your configuration.</p>
            </div>
        </section>
        """, unsafe_allow_html=True)
        return
    
    # Generation Section
    st.markdown('<div id="generate-section-start" style="display: none;"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h2 class="section-title">Enter Recipe Details</h2>', unsafe_allow_html=True)
        
        recipe_title = st.text_input(
            "Recipe Title",
            placeholder="e.g., Chocolate Chip Cookies",
            label_visibility="collapsed"
        )
        
        ingredients_input = st.text_area(
            "Ingredients (comma-separated)",
            placeholder="e.g., butter, sugar, eggs, flour, chocolate chips, vanilla extract",
            height=150,
            label_visibility="collapsed"
        )
        
        # Generation Settings (in main content, not sidebar)
        st.markdown('<h3 class="settings-title">⚙️ Generation Settings</h3>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, help="Higher = more creative")
            top_k = st.slider("Top K", 10, 100, 50, 10)
            max_length = st.slider("Max Length", 256, 1024, 512, 64)
        
        with col_b:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
            num_variations = st.selectbox("Variations", [1, 2, 3], index=0)
            st.markdown(f'<div class="device-info">💻 Device: {DEVICE.upper()}</div>', unsafe_allow_html=True)
        
        generate_btn = st.button("🚀 Generate Recipe", type="primary", use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-title">Generated Recipe</h2>', unsafe_allow_html=True)
        
        if generate_btn:
            if not recipe_title or not ingredients_input:
                st.markdown("""
                <div class="placeholder">
                    <div class="placeholder-icon">⚠️</div>
                    <p>Please enter both recipe title and ingredients!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner('🔮 Generating your recipe...'):
                    generated_directions = generate_recipe(
                        title=recipe_title,
                        ingredients=ingredients_input,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_length=max_length,
                        num_samples=num_variations
                    )
                    
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    
                    for idx, directions in enumerate(generated_directions):
                        if num_variations > 1:
                            st.markdown(f'<h3 class="variation-title">📋 Variation {idx + 1}</h3>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                        
                        st.markdown(f"### 🍽️ {recipe_title}")
                        
                        st.markdown("**Ingredients:**")
                        ingredients_list = [ing.strip() for ing in ingredients_input.split(',')]
                        
                        ingredients_html = "".join([f'<span class="ingredient-tag">{ing}</span>' for ing in ingredients_list])
                        st.markdown(ingredients_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        st.markdown("**Directions:**")
                        steps = parse_directions(directions)
                        
                        if steps:
                            for i, step in enumerate(steps, 1):
                                st.markdown(f"""
                                    <div class="step-item">
                                        <span class="step-number">{i}</span>
                                        <span class="step-text">{step}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write(directions)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
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
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-icon">📝</div>
                <p>Enter recipe details and click 'Generate Recipe' to see the magic!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <section class="about-section" id="about">
        <div class="about-container">
            <h2 class="section-title centered">About AI Recipe Generator</h2>
            <div class="about-grid">
                <div class="about-card">
                    <div class="about-icon">🤖</div>
                    <h3>GPT-2 Powered</h3>
                    <p>Fine-tuned GPT-2 language model trained on thousands of recipes to generate authentic cooking directions.</p>
                </div>
                <div class="about-card">
                    <div class="about-icon">⚡</div>
                    <h3>Instant Results</h3>
                    <p>Get complete recipe directions in seconds with customizable generation parameters for perfect results.</p>
                </div>
                <div class="about-card">
                    <div class="about-icon">🎨</div>
                    <h3>Creative Freedom</h3>
                    <p>Experiment with any combination of ingredients and titles to discover unique recipe variations.</p>
                </div>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <footer class="footer">
        <p>Built with ❤️ using Streamlit and PyTorch | Fine-tuned GPT-2 Model</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
