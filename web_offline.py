import streamlit as st
import torch
from streamlit_chat import message
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from src.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
    
model_info = {
    "SeedGPT-V3" : {
        "name": "SeedGPT-V3",
        "params": "22M parameters",
        "dataset": "Trained on lmsys chat english dataset",
        "dataset_link" : "https://shorturl.at/PZANz",
        "purpose": "Fine-tuned for chat style conversation"
    },
    "SeedGPT-V2" : {
        "name": "SeedGPT-V2",
        "params": "22M parameters",
        "dataset": "Trained on Tinystories & stories dataset",
        "dataset_link" : "https://shorturl.at/F1ZvX & https://shorturl.at/ndPa4",
        "purpose": "Generate text based on input text"
    },
     "SeedGPT-V1" : {
        "name": "SeedGPT-V1",
        "params": "22M parameters",
        "dataset": "Trained on refined bookcorpus dataset",
        "dataset_link" : "https://shorturl.at/FezgK",
        "purpose": "Generate text based on input text"
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
temp = st.sidebar.slider(label="🌡️ Temperature",min_value=0.2,max_value=1.0,step=0.05,value=0.7,
                         disabled=st.session_state.get("is_generating", False))
st.sidebar.markdown("</br>",unsafe_allow_html=True)
model_type = st.sidebar.selectbox("🧠 Select model",options=model_info.keys(),
                                  disabled=st.session_state.get("is_generating", False))
model_details = model_info[model_type]

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict
    
# -------- Load tokenizer & model in session_state --------
if "model" not in st.session_state or st.session_state.get("model_type") != model_type:
    st.session_state["is_generating"] = True
    with st.spinner(f"🔄 Loading model **{model_type}**... Please wait! ⚙️\n🚫 Do not change any settings or enter a prompt now."):
        try:
            if model_type != "SeedGPT-V3":
                st.session_state["tokenizer"] = Tokenizer.from_file("./saved_tokenizer/tokenizer.json")
                st.session_state["model"] =  Transformer(vocab_size=2000,
                                                         emb_size=512,
                                                         max_seq=256,
                                                         num_head=4,
                                                         num_block=6).to(device=device)
                model_param = torch.load(f"./models/{model_type}.pt",map_location=torch.device(device=device))
                st.session_state["model"].load_state_dict(remove_module_prefix(model_param["model_state_dict"]))
            else:
                st.session_state["tokenizer"] = AutoTokenizer.from_pretrained("./models/SeedGPT-V3/")
                checkpoint = torch.load("./models/SeedGPT-V3/SeedGPT-V3.bin", map_location=device)
                config = checkpoint["config"]
                st.session_state["model"] = Transformer(
                    vocab_size=config["vocab_size"],
                    emb_size=config["emb_size"],
                    max_seq=config["max_seq"],
                    num_head=config["num_head"],
                    num_block=config["num_block"]
                    ).to(device)
                st.session_state["model"].load_state_dict(checkpoint["model_state_dict"])
            st.session_state["model"].to(device)
            st.session_state["model_type"] = model_type
        finally:
            st.session_state["is_generating"] = False


tokenizer = st.session_state["tokenizer"]
model = st.session_state["model"]

with st.sidebar.expander("📄 Model Info", expanded=False):
    st.markdown(f"""
    **🧬 Model Name**: {model_details['name']}  
    **📊 Parameters**: {model_details['params']}  
    **📚 Dataset**: {model_details['dataset']}  
    **📖 Dataset Link**: {model_details['dataset_link']}  
    **🎯 Purpose**: {model_details['purpose']}
    """)

st.sidebar.markdown("</br>",unsafe_allow_html=True)
max_num_tokens = st.sidebar.slider(label="🔠 Max Tokens",min_value=10,max_value=4096,value=100,step=1,
                                   disabled=st.session_state.get("is_generating", False))

if st.sidebar.button("🧹 Clear Chat",disabled=st.session_state.get("is_generating", False)):
    st.session_state["messages"] = []
    st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for idx,msg in enumerate(st.session_state["messages"]):
    is_user = msg["role"] == "user"
    if msg["role"] == "assistant":
        avatar = "bottts"
        seed = "Aneka"
    else:
        avatar = "miniavs"
        seed = "solid"
    message(msg["content"],is_user=is_user,key=str(idx),avatar_style=avatar,seed=seed)


tokenizer.chat_template = """
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
if  not st.session_state["is_generating"]:
    prompt = st.chat_input("💬 Ask SeedGPT ...", max_chars=100)
else:
    st.chat_input("💬 Ask SeedGPT ...", max_chars=100, disabled=True)
    #st.info("⏳ Please wait! SeedGPT is generating a response...")

# If new prompt submitted, store it & rerun
if prompt:
    st.session_state["pending_prompt"] = prompt
    st.session_state["is_generating"] = True
    st.rerun()

# If there is a pending prompt and we are generating, do the generation
if st.session_state.get("is_generating", False) and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")  # Get and remove it
    st.session_state["messages"].append({"role":"user","content":prompt})
    try:
        if model_details['name'] == "SeedGPT-V3":
            chat = [{"role": "user", "content": prompt}]
            input_txt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_txt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with st.spinner("🌱 SeedGPT is thinking..."):
                with torch.no_grad():
                    output = model.generate(inputs["input_ids"], max_tokens=max_num_tokens, temp=temp)
                generated = output[0][inputs["input_ids"].shape[1]:]
                output_txt = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            tokens = tokenizer.encode(prompt)
            input_tokens = torch.tensor(tokens.ids,dtype=torch.long)[None,:]
            input_tokens = input_tokens.to(device)
            with st.spinner("🌱 SeedGPT is thinking..."):
                with torch.no_grad():
                    response = model.generate(input_tokens, max_num_tokens, temp)
            output_txt = tokenizer.decode(response[0].tolist(), skip_special_tokens=True)
            output_txt = output_txt.replace("</S>", "").strip()

        st.session_state["messages"].append({"role":"assistant","content":output_txt})
    finally:
        st.session_state["is_generating"] = False
        st.rerun()
