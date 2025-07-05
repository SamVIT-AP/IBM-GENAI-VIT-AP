
import streamlit as st
from transformers import pipeline, set_seed

st.set_page_config(page_title="Free AI Story Generator", layout="centered")
st.title("📖 Free AI Story Generator")
st.markdown("Enter a story prompt and generate a short creative story using GPT-Neo (125M).")

prompt = st.text_input("Enter a story prompt:", "A robot discovers human emotions")
max_length = st.slider("Story length (words)", 50, 300, 120)

if st.button("Generate Story"):
    with st.spinner("Generating story..."):
        try:
            generator = pipeline(
                "text-generation",
                model="EleutherAI/gpt-neo-125M",
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=50256
            )
            set_seed(42)
            output = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1
            )[0]["generated_text"]

            st.success("Here's your story:")
            st.text_area("Generated Story", output, height=300)
            st.download_button("📥 Download Story", output, file_name="story.txt")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
