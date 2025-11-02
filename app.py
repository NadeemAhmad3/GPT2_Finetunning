import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Page configuration
st.set_page_config(
    page_title="AI Recipe Generator",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.recipe-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin: 20px 0;
}
.ingredient-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 15px 0;
}
.title-text {
    font-size: 2.5em;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Title
st.markdown('<p class="title-text">🍳 AI Recipe Generator</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Generate delicious recipes using fine-tuned GPT-2</p>", unsafe_allow_html=True)

# Sidebar for model loading
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    # Model path input
    model_path = st.text_input(
        "Model Directory Path",
        value="./recipe-gpt2-finetuned",
        help="Enter the path to your fine-tuned model directory"
    )
    
    # Load model button
    if st.button("📂 Load Model", use_container_width=True):
        if not os.path.exists(model_path):
            st.error(f"❌ Directory not found: {model_path}")
        else:
            with st.spinner("Loading model... Please wait..."):
                try:
                    # Load model and tokenizer
                    model = GPT2LMHeadModel.from_pretrained(model_path)
                    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                    
                    # Set pad token
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    
                    st.success("✅ Model loaded successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_loaded = False
    
    # Model status
    st.markdown("---")
    if st.session_state.model_loaded:
        st.success("🟢 Model Ready")
        if st.button("🗑️ Unload Model"):
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.model_loaded = False
            st.rerun()
    else:
        st.warning("🔴 No Model Loaded")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Enter your model path
    2. Click 'Load Model'
    3. Choose input method
    4. Generate recipes!
    """)

# Main content
if not st.session_state.model_loaded:
    # Show instructions when no model is loaded
    st.info("👈 Please load your fine-tuned model from the sidebar to start generating recipes!")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Load Model")
        st.write("Enter the path to your trained model and click 'Load Model'")
    
    with col2:
        st.markdown("#### 2️⃣ Choose Input")
        st.write("Enter a recipe title or list of ingredients")
    
    with col3:
        st.markdown("#### 3️⃣ Generate")
        st.write("Click generate and get your AI-created recipe!")

else:
    # Recipe generation interface
    st.markdown("---")
    
    # Input method selection
    st.subheader("🎨 Create Your Recipe")
    
    input_method = st.radio(
        "Choose how to generate your recipe:",
        ["🏷️ Recipe Title", "🥕 Ingredients List", "✍️ Custom Prompt"],
        horizontal=True
    )
    
    prompt = ""
    
    if input_method == "🏷️ Recipe Title":
        st.markdown("### Enter Recipe Name")
        recipe_title = st.text_input(
            "What would you like to cook?",
            placeholder="e.g., Chocolate Chip Cookies, Chicken Curry, Caesar Salad...",
            label_visibility="collapsed"
        )
        if recipe_title:
            prompt = f"Recipe: {recipe_title}"
    
    elif input_method == "🥕 Ingredients List":
        st.markdown("### Enter Your Ingredients")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ingredients_input = st.text_area(
                "List your ingredients (one per line):",
                height=200,
                placeholder="chicken breast\nonions\ngarlic\ntomatoes\nolive oil\nsalt\npepper",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**💡 Tips:**")
            st.markdown("""
            - Enter one ingredient per line
            - Be specific (e.g., "chicken breast" not just "chicken")
            - Include quantities if you want
            - More ingredients = more detailed recipe
            """)
        
        if ingredients_input:
            ingredients = [ing.strip() for ing in ingredients_input.split('\n') if ing.strip()]
            if ingredients:
                st.markdown(f"**Selected ingredients:** {', '.join(ingredients)}")
                ingredients_str = '\n'.join(ingredients)
                prompt = f"Recipe: Dish\n\nIngredients:\n{ingredients_str}\n\nInstructions:"
    
    else:  # Custom Prompt
        st.markdown("### Enter Custom Prompt")
        prompt = st.text_area(
            "Write your custom prompt:",
            height=200,
            placeholder="Recipe: Spicy Thai Curry\n\nIngredients:\ncoconut milk\nred curry paste\nchicken\n\nInstructions:",
            label_visibility="collapsed"
        )
    
    # Generation parameters in expander
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider(
                "Maximum Length",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="Maximum number of tokens to generate"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
        
        with col2:
            top_k = st.slider(
                "Top K",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Number of top tokens to consider"
            )
            
            top_p = st.slider(
                "Top P (Nucleus Sampling)",
                min_value=0.5,
                max_value=1.0,
                value=0.95,
                step=0.05,
                help="Cumulative probability threshold"
            )
        
        num_recipes = st.slider(
            "Number of Recipes to Generate",
            min_value=1,
            max_value=5,
            value=1,
            help="Generate multiple recipe variations"
        )
    
    # Generate button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🎨 Generate Recipe",
            type="primary",
            use_container_width=True
        )
    
    # Generation
    if generate_button:
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ Please provide some input to generate a recipe!")
        else:
            with st.spinner("👨‍🍳 Creating your recipe... This may take a moment..."):
                try:
                    # Prepare input
                    input_ids = st.session_state.tokenizer.encode(
                        prompt,
                        return_tensors='pt'
                    )
                    
                    # Generate
                    with torch.no_grad():
                        outputs = st.session_state.model.generate(
                            input_ids,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=True,
                            num_return_sequences=num_recipes,
                            pad_token_id=st.session_state.tokenizer.eos_token_id
                        )
                    
                    # Decode outputs
                    generated_recipes = []
                    for output in outputs:
                        recipe = st.session_state.tokenizer.decode(
                            output,
                            skip_special_tokens=True
                        )
                        generated_recipes.append(recipe)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🍽️ Your Generated Recipe(s)")
                    
                    for i, recipe in enumerate(generated_recipes, 1):
                        with st.container():
                            if num_recipes > 1:
                                st.markdown(f"### Recipe Variation {i}")
                            
                            # Display recipe in a nice box
                            st.markdown(
                                f'<div class="ingredient-box"><pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">{recipe}</pre></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Download button
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col2:
                                st.download_button(
                                    label=f"💾 Download Recipe {i}",
                                    data=recipe,
                                    file_name=f"recipe_{i}.txt",
                                    mime="text/plain",
                                    key=f"download_{i}",
                                    use_container_width=True
                                )
                            
                            if i < len(generated_recipes):
                                st.markdown("---")
                    
                    st.success("✅ Recipe(s) generated successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error generating recipe: {str(e)}")
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Made with ❤️ using Streamlit & GPT-2 | "
    "Fine-tuned for Recipe Generation</p>",
    unsafe_allow_html=True
)
