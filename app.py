import streamlit as st
import openai
from PIL import Image
import base64
import io
import time

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_api_key():
    # Directly define your API key here (replace with your actual key)
    return "3a65848d-5b28-46c9-a5ec-14b015ed9f9d"

def initialize_sambanova():
    api_key = get_api_key()
    return openai.OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")

# Set up the Streamlit App
st.set_page_config(
    page_title="SambaNova Medical Image Diagnosis Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for API Key Input
with st.sidebar:
    st.title("Setup")
    st.markdown("Get your SambaNova API key [here](https://sambanova.ai)")
    api_key_input = st.text_input("SAMBANOVA CLOUD API KEY", type="password")
    save_credentials = st.button("Save Credentials")

    if save_credentials and api_key_input:
        st.session_state["sambanova_api_key"] = api_key_input
        st.success("API Key saved successfully!")

# Initialize model with user-provided API key
if "sambanova_api_key" in st.session_state:
    client = openai.OpenAI(api_key=st.session_state["sambanova_api_key"], base_url="https://api.sambanova.ai/v1")
else:
    st.warning("Please enter your API key in the sidebar.")
    st.stop()

# Title Section
st.markdown("<h1 style='text-align: center; color: #f4a261;'>SambaNova <span style='color: white;'>Medical Image Diagnosis Assistant</span></h1>", unsafe_allow_html=True)

# Sidebar for image upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        encoded_image = encode_image(image)
    else:
        encoded_image = None

# Main chat interface
chat_placeholder = st.container()

with chat_placeholder:
    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input handling
prompt = st.chat_input("Ask questions about your data")

if prompt:
    inputs = [{"type": "text", "text": prompt}]
    
    if encoded_image:
        inputs.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Generate and display response
    with st.spinner('Generating response...'):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="Llama-3.2-11B-Vision-Instruct",
                messages=[{"role": "user", "content": inputs}],
                temperature=0.1,
                top_p=0.1
            )
            end_time = time.time()
            time_taken = end_time - start_time
            
            reply = response.choices[0].message.content
            
            with chat_placeholder:
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    st.markdown(f"_Response time: {time_taken:.2f} seconds_")
                    
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif uploaded_file:
    st.warning("Please enter a text query to accompany the image.")