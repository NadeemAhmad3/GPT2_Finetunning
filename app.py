import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import time
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="GPT2 Model Tester",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Header
st.markdown('<div class="main-header">🤖 T5 Model Testing Interface</div>', unsafe_allow_html=True)

# Sidebar - Model Configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Device info
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Using: **{device_name}**")
    
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    
    # Model loading section
    st.subheader("📂 Load Model")
    
    model_source = st.radio(
        "Model Source:",
        ["Local Path", "Hugging Face Hub"],
        help="Choose where to load your model from"
    )
    
    if model_source == "Local Path":
        model_path = st.text_input(
            "Model Directory Path:",
            placeholder="./my_finetuned_t5",
            help="Enter the path to your fine-tuned model directory"
        )
    else:
        model_path = st.text_input(
            "Model Name:",
            placeholder="t5-small",
            help="Enter the Hugging Face model name"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Model"):
            if model_path:
                with st.spinner("Loading model..."):
                    try:
                        st.session_state.tokenizer = T5Tokenizer.from_pretrained(model_path)
                        st.session_state.model = T5ForConditionalGeneration.from_pretrained(model_path)
                        st.session_state.model.to(st.session_state.device)
                        st.session_state.model.eval()
                        st.success("✅ Model loaded successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
            else:
                st.warning("⚠️ Please enter a model path")
    
    with col2:
        if st.button("🗑️ Clear Model"):
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.success("Model cleared!")
            time.sleep(1)
            st.rerun()
    
    # Model status
    st.divider()
    if st.session_state.model is not None:
        st.success("✅ Model Status: **Loaded**")
        
        # Model info
        with st.expander("📊 Model Information"):
            num_params = sum(p.numel() for p in st.session_state.model.parameters())
            st.write(f"**Total Parameters:** {num_params:,}")
            st.write(f"**Model Type:** {st.session_state.model.config.model_type}")
            st.write(f"**Vocab Size:** {st.session_state.model.config.vocab_size:,}")
    else:
        st.warning("⚠️ Model Status: **Not Loaded**")
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared!")
        time.sleep(1)
        st.rerun()

# Main content area
if st.session_state.model is None:
    st.warning("⚠️ Please load a model from the sidebar to begin testing.")
    
    with st.expander("📖 How to use this app"):
        st.markdown("""
        ### Getting Started
        1. **Load your model** from the sidebar using either:
           - Local path to your fine-tuned model directory
           - Hugging Face model name
        2. **Enter your input text** in the text area
        3. **Configure generation parameters** (optional)
        4. **Click Generate** to test your model
        5. **View results** and download history
        
        ### Tips
        - Use the prefix format your model was trained on (e.g., "summarize: ", "translate English to French: ")
        - Adjust generation parameters to control output quality
        - Save your test history for later analysis
        """)
else:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🧪 Single Test", "📝 Batch Test", "📊 History"])
    
    # Tab 1: Single Test
    with tab1:
        st.header("Single Input Test")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_text = st.text_area(
                "Input Text:",
                height=150,
                placeholder="Enter your text here... (e.g., 'summarize: Your text to summarize')",
                help="Enter the text you want to test. Include task prefix if your model was trained with one."
            )
            
            # Quick templates
            st.write("**Quick Templates:**")
            template_cols = st.columns(4)
            with template_cols[0]:
                if st.button("Summarize"):
                    st.session_state.template_text = "summarize: "
            with template_cols[1]:
                if st.button("Translate"):
                    st.session_state.template_text = "translate English to French: "
            with template_cols[2]:
                if st.button("Question"):
                    st.session_state.template_text = "question: "
            with template_cols[3]:
                if st.button("Custom"):
                    st.session_state.template_text = ""
        
        with col2:
            st.subheader("Generation Parameters")
            
            max_length = st.slider(
                "Max Length:",
                min_value=10,
                max_value=512,
                value=128,
                step=10,
                help="Maximum length of generated text"
            )
            
            num_beams = st.slider(
                "Num Beams:",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of beams for beam search"
            )
            
            temperature = st.slider(
                "Temperature:",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Sampling temperature (lower = more deterministic)"
            )
            
            top_k = st.slider(
                "Top-K:",
                min_value=0,
                max_value=100,
                value=50,
                help="Top-K sampling parameter"
            )
            
            top_p = st.slider(
                "Top-P:",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.05,
                help="Nucleus sampling parameter"
            )
            
            repetition_penalty = st.slider(
                "Repetition Penalty:",
                min_value=1.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Penalty for repeating tokens"
            )
        
        if st.button("🚀 Generate", type="primary"):
            if input_text.strip():
                with st.spinner("Generating..."):
                    try:
                        start_time = time.time()
                        
                        # Tokenize
                        inputs = st.session_state.tokenizer(
                            input_text,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True
                        ).to(st.session_state.device)
                        
                        # Generate
                        with torch.no_grad():
                            outputs = st.session_state.model.generate(
                                **inputs,
                                max_length=max_length,
                                num_beams=num_beams,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                early_stopping=True
                            )
                        
                        # Decode
                        output_text = st.session_state.tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                        
                        generation_time = time.time() - start_time
                        
                        # Display results
                        st.success("✅ Generation Complete!")
                        
                        st.subheader("📤 Output:")
                        st.info(output_text)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Input Tokens", len(inputs['input_ids'][0]))
                        with col2:
                            st.metric("Output Tokens", len(outputs[0]))
                        with col3:
                            st.metric("Time (s)", f"{generation_time:.2f}")
                        
                        # Save to history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'input': input_text,
                            'output': output_text,
                            'max_length': max_length,
                            'num_beams': num_beams,
                            'temperature': temperature,
                            'generation_time': generation_time
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error during generation: {str(e)}")
            else:
                st.warning("⚠️ Please enter some input text")
    
    # Tab 2: Batch Test
    with tab2:
        st.header("Batch Input Test")
        
        st.write("Test multiple inputs at once. Enter one input per line.")
        
        batch_input = st.text_area(
            "Batch Inputs (one per line):",
            height=200,
            placeholder="summarize: First text to test\nsummarize: Second text to test\nsummarize: Third text to test"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            batch_max_length = st.number_input(
                "Max Length:",
                min_value=10,
                max_value=512,
                value=128
            )
            
            batch_num_beams = st.number_input(
                "Num Beams:",
                min_value=1,
                max_value=10,
                value=4
            )
        
        if st.button("🚀 Generate Batch", type="primary"):
            if batch_input.strip():
                inputs_list = [line.strip() for line in batch_input.split('\n') if line.strip()]
                
                if inputs_list:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, input_text in enumerate(inputs_list):
                        status_text.text(f"Processing {idx + 1}/{len(inputs_list)}...")
                        
                        try:
                            inputs = st.session_state.tokenizer(
                                input_text,
                                return_tensors="pt",
                                max_length=512,
                                truncation=True
                            ).to(st.session_state.device)
                            
                            with torch.no_grad():
                                outputs = st.session_state.model.generate(
                                    **inputs,
                                    max_length=batch_max_length,
                                    num_beams=batch_num_beams,
                                    early_stopping=True
                                )
                            
                            output_text = st.session_state.tokenizer.decode(
                                outputs[0],
                                skip_special_tokens=True
                            )
                            
                            results.append({
                                'Input': input_text,
                                'Output': output_text
                            })
                            
                        except Exception as e:
                            results.append({
                                'Input': input_text,
                                'Output': f"Error: {str(e)}"
                            })
                        
                        progress_bar.progress((idx + 1) / len(inputs_list))
                    
                    status_text.text("✅ Batch processing complete!")
                    
                    # Display results
                    st.subheader("📊 Batch Results")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "batch_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("⚠️ No valid inputs found")
            else:
                st.warning("⚠️ Please enter batch inputs")
    
    # Tab 3: History
    with tab3:
        st.header("Test History")
        
        if st.session_state.history:
            st.write(f"**Total Tests:** {len(st.session_state.history)}")
            
            # Display history
            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Test {len(st.session_state.history) - idx} - {entry['timestamp']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Input:**")
                        st.text(entry['input'])
                    
                    with col2:
                        st.write("**Output:**")
                        st.text(entry['output'])
                    
                    st.write("**Parameters:**")
                    param_cols = st.columns(4)
                    with param_cols[0]:
                        st.metric("Max Length", entry['max_length'])
                    with param_cols[1]:
                        st.metric("Num Beams", entry['num_beams'])
                    with param_cols[2]:
                        st.metric("Temperature", entry['temperature'])
                    with param_cols[3]:
                        st.metric("Time (s)", f"{entry['generation_time']:.2f}")
            
            # Download history
            if st.button("📥 Download History (JSON)"):
                json_str = json.dumps(st.session_state.history, indent=2)
                st.download_button(
                    "Download",
                    json_str,
                    "test_history.json",
                    "application/json"
                )
        else:
            st.info("📭 No test history yet. Run some tests to see them here!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 GPT2 Model Testing Interface | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
