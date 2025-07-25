import streamlit as st
import torch
from streamlit_chat import message
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from src.convert_to_hf_model import HFTransformerConfig, HFTransformerModel  # custom config and model

# Register custom config and model
CONFIG_MAPPING.register("hf_transformer", HFTransformerConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(HFTransformerConfig, HFTransformerModel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []
        
model_info = {
    "SeedGPT-V3" : {
        "name": "SeedGPT-V3",
        "params": "22M parameters",
        "dataset": "Trained on lmsys chat english dataset",
        "dataset_link" : "https://shorturl.at/PZANz",
        "purpose": "Fine-tuned for chat style conversation",
        "repo": "huggingface.co/singhsumony2j/SeedGPT-V3"
    },
    "SeedGPT-V2" : {
        "name": "SeedGPT-V2",
        "params": "22M parameters",
        "dataset": "Trained on Tinystories & stories dataset",
        "dataset_link" : "https://shorturl.at/F1ZvX & https://shorturl.at/ndPa4",
        "purpose": "Generate text based on input text",
        "repo": "huggingface.co/singhsumony2j/SeedGPT-V2"
    },
     "SeedGPT-V1" : {
        "name": "SeedGPT-V1",
        "params": "22M parameters",
        "dataset": "Trained on refined bookcorpus dataset",
        "dataset_link" : "https://shorturl.at/FezgK",
        "purpose": "Generate text based on input text",
        "repo": "huggingface.co/singhsumony2j/SeedGPT-V1"
    }
}

st.set_page_config(page_title="SeedGPT",page_icon=":deciduous_tree:",layout="wide")

# ----- Custom Styled Title -----
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2rem; color: #2e7d32;'>🌱 SeedGPT - A Small Language Model</h1>
        <hr style='border-top: 2px solid #81c784; width: 60%; margin: auto;'/>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color: #2e7d32;'>🛠️ Settings</h3>", unsafe_allow_html=True)

temp = st.sidebar.slider(label="🌡️ Temperature",min_value=0.2,max_value=1.4,step=0.05,value=0.7,
                         disabled=st.session_state.get("is_generating", False))

st.sidebar.markdown("</br>",unsafe_allow_html=True)

max_num_tokens = st.sidebar.slider(label="🔠 Max Tokens",min_value=10,max_value=4096,value=100,step=1,disabled=st.session_state.get("is_generating", False))

st.sidebar.markdown("</br>",unsafe_allow_html=True)

if st.sidebar.button("🧹 Clear Chat",disabled=st.session_state.get("is_generating", False)):
    st.session_state["messages"] = []
    st.rerun()

st.sidebar.markdown("</br>",unsafe_allow_html=True)

for idx,msg in enumerate(st.session_state["messages"]):
    is_user = msg["role"] == "user"
    if msg["role"] == "assistant":
        avatar = "bottts"
        seed = "Aneka"
    else:
        avatar = "miniavs"
        seed = "solid"
    message(msg["content"],is_user=is_user,key=str(idx),avatar_style=avatar,seed=seed)

if "model_type" not in st.session_state:
    st.session_state["model_type"] = None
    
# Get the selected model type from the selectbox
selected_model_type = st.sidebar.selectbox(
    "🧠 Select model", options=list(model_info.keys()),
    disabled=st.session_state.get("is_generating", False)
)
model_details = model_info[selected_model_type]

# --- MODEL RELOAD LOGIC ---
# 'model_type' in session_state means model is loaded
# It should only ever represent the *loaded* model type

needs_loading = (
    ("model" not in st.session_state)
    or ("model_type" not in st.session_state)
    or (st.session_state["model_type"] != selected_model_type)
)

if needs_loading and not st.session_state.get("is_generating", False):
    st.session_state["is_generating"] = True
    st.rerun()


if needs_loading and st.session_state.get("is_generating", False):
    with st.spinner(f"🫸 🔄 Please wait while we set things up ⚙️",show_time=True): 
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(f"singhsumony2j/{selected_model_type}")
        st.session_state["model"] = AutoModelForCausalLM.from_pretrained(
            f"singhsumony2j/{selected_model_type}",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        st.session_state["model_type"] = selected_model_type  # Mark model as loaded
        st.session_state["is_generating"] = False
        st.session_state.pop("pending_prompt", None)
        st.rerun()

# STOP execution here if model not yet ready
if ("model" not in st.session_state) or ("tokenizer" not in st.session_state):
    st.stop()


if (
    "pending_prompt" in st.session_state and 
    not st.session_state.get("is_generating", False)
):
    st.session_state.pop("pending_prompt", None)

tokenizer = st.session_state["tokenizer"]
model = st.session_state["model"]
model_details = model_info[st.session_state["model_type"]]

with st.sidebar.expander("📄 Model Info", expanded=False):
    st.markdown(f"""
    **🧬 Model Name**: {model_details['name']}  
    **📊 Parameters**: {model_details['params']}  
    **📚 Dataset**: {model_details['dataset']}  
    **📖 Dataset Link**: {model_details['dataset_link']}  
    **🎯 Purpose**: {model_details['purpose']}  
    **🔗 HF Repo**: [{model_details['repo']}](https://{model_details['repo']})
    """)

st.session_state["tokenizer"].chat_template = """
{% for message in messages %}
{% if message["role"] == "user" %}
<S>user: {{ message["content"] }}</S>
{% elif message["role"] == "assistant" %}
<S>assistant: {{ message["content"] }}</S>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<S>assistant:
{% endif %}
"""

# This handles when user hits Enter
prompt = None
if not st.session_state["is_generating"]:
    prompt = st.chat_input("💬 Ask SeedGPT ...", max_chars=100)
else:
    st.chat_input("💬 Ask SeedGPT ...", max_chars=100, disabled=True)
    if st.session_state.get("pending_prompt") and "model" not in st.session_state:
        st.warning("⚠️ You submitted a prompt before the model finished loading. This may result in incomplete or incorrect output. Please wait for the model to fully load before submitting a prompt.")

# If new prompt submitted, store it & rerun
if prompt:
    if ("model" not in st.session_state) or ("tokenizer" not in st.session_state):
        st.warning("⚠️ Model is still loading! Please wait until loading is complete before submitting your query.")
        st.stop()
    # Otherwise, process as normal
    st.session_state["pending_prompt"] = prompt
    st.session_state["is_generating"] = True
    st.rerun()

# If there is a pending prompt and we are generating, do the generation

if (
    st.session_state.get("is_generating", False)
    and st.session_state.get("pending_prompt")
    and ("model" in st.session_state)
    and ("tokenizer" in st.session_state)
):
    
    prompt = st.session_state.pop("pending_prompt")
    st.session_state["messages"].append({"role": "user", "content": prompt})
    tokenizer = st.session_state["tokenizer"]
    model = st.session_state["model"]
    try:
        if model_details['name'] == "SeedGPT-V3":
            chat = [{"role": "user", "content": prompt}]
            input_txt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_txt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with st.spinner("🌱 SeedGPT is thinking...", show_time=True):
                with torch.no_grad():
                    output = model.generate(inputs["input_ids"], max_tokens=max_num_tokens, temp=temp)
                generated = output[0][inputs["input_ids"].shape[1]:]
                output_txt = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            tokens = tokenizer(prompt)
            input_tokens = torch.tensor(tokens.input_ids,dtype=torch.long)[None,:]
            input_tokens = input_tokens.to(device)
            with st.spinner("🌱 SeedGPT is thinking...", show_time=True):
                with torch.no_grad():
                    response = model.generate(input_tokens, max_num_tokens, temp)
                output_txt = tokenizer.decode(response[0].tolist(), skip_special_tokens=True)
                output_txt = output_txt.replace("</S>", "").strip()

        st.session_state["messages"].append({"role":"assistant","content":output_txt})
    finally:
        st.session_state["is_generating"] = False
        st.rerun()
