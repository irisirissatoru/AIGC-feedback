import uuid
import openai
import streamlit as st
from PIL import Image
import io
import random
import requests
from openai import OpenAI
import os

# Simulated scoring function
def score_image(image: Image.Image) -> float:
    score = round(random.uniform(5.0, 9.5), 2)
    st.info(f"authentic score: {score}")
    return score

# qwen
client = OpenAI(
    api_key=st.secrets["DASHSCOPE_API_KEY"],  # 从 Streamlit secrets.toml 读取
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_feedback(prompt: str, score: float) -> str:
    system_msg = "You are a professional image prompt critic and must provide clear, confident feedback without hedging language."
    user_msg = f'''
Prompt: {prompt}
Score: {score}/10
 
Please do the following in English:

1. **Directly describe** what the image looks like (no phrases like "might show", "may be", etc.)
2. **Critically analyze** flaws in visual quality or how the prompt affected them — be specific
3. **Propose a clearly better prompt** to improve visual fidelity and detail

Avoid hedging expressions like "may", "might", "could", or "possibly". Be confident and specific.
'''

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            # 需要时可加这行避免报错（看你是否使用 Qwen3）
            # extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error from Qwen API: {e}"


# Streamlit UI
st.title("🎨Gen-image Scorer")

prompt = st.text_area("Enter your text prompt here:")
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if st.button("Evaluate") and prompt and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    score = score_image(image)
    feedback = get_feedback(prompt, score)

    st.image(image, caption=f"Image Score: {score}/10", use_container_width=True)
    st.markdown("### 📝 GPT Feedback and Prompt Suggestions")
    st.write(feedback)
else:
    st.info("Please upload an image and enter a prompt to begin.")
