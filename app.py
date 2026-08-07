"""
app.py
======
Streamlit UI for GPT-2 text generation.
"""

import streamlit as st

from config import CONFIG


def load_generator():
    """Load generator with graceful fallback if model doesn't exist."""
    from inference import TextGenerator, GenerationParams
    
    if not CONFIG.paths.saved_model_dir.exists():
        return None
    try:
        return TextGenerator(CONFIG.paths.saved_model_dir, CONFIG.device)
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None


def build_params(
    strategy: str,
    max_new_tokens: int,
    num_return_sequences: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    num_beams: int,
):
    from inference import GenerationParams
    
    if strategy == "Greedy":
        return GenerationParams(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, num_return_sequences=1)
    if strategy == "Beam Search":
        return GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
        )
    if strategy == "Top-k Sampling":
        return GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
    if strategy == "Top-p (Nucleus) Sampling":
        return GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
    if strategy == "Temperature Sampling":
        return GenerationParams(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
    return GenerationParams(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
    )


def main() -> None:
    st.set_page_config(page_title="GPT-2 Quote Generator", page_icon="✍️")
    st.title("✍️ GPT-2 Quote Generator")

    generator = load_generator()
    
    # Show status if model is not loaded
    if generator is None:
        st.warning("⚠️ No trained model found. Train a model first or check the saved_model/ directory.")

    prompt = st.text_area("Prompt", value="Life is", height=80)

    strategy = st.selectbox(
        "Decoding strategy",
        [
            "Greedy",
            "Beam Search",
            "Top-k Sampling",
            "Top-p (Nucleus) Sampling",
            "Temperature Sampling",
            "Custom (all params)",
        ],
    )

    with st.sidebar:
        st.header("Generation Settings")
        max_new_tokens = st.slider("Max new tokens", 10, 200, 60, step=10)
        num_return_sequences = st.slider("Number of outputs", 1, 5, 1)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1)
        top_k = st.slider("Top-k", 0, 200, 50, step=5)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.95, step=0.05)
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, step=0.05)
        num_beams = st.slider("Beam count (beam search only)", 1, 10, 5)

    if st.button("Generate", type="primary", disabled=(generator is None)):
        if not prompt.strip():
            st.error("Enter a prompt first.")
        else:
            params = build_params(
                strategy,
                max_new_tokens,
                num_return_sequences,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                num_beams,
            )
            try:
                with st.spinner("Generating..."):
                    outputs = generator.generate(prompt, params)
                st.session_state["outputs"] = outputs
            except Exception as e:
                st.error(f"Generation failed: {e}")

    if "outputs" in st.session_state:
        st.subheader("Generated Text")
        combined = ""
        for i, text in enumerate(st.session_state["outputs"], start=1):
            st.markdown(f"**Output {i}:**")
            st.write(text)
            combined += f"Output {i}:\n{text}\n\n"

        st.download_button(
            label="Download as .txt",
            data=combined,
            file_name="generated_quotes.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
