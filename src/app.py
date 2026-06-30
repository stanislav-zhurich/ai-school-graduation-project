"""``run-app`` entry point: a minimal Streamlit chat over the wine agent."""
from __future__ import annotations

import sys

import streamlit as st

from agent import run_agent


def render() -> None:
    """Render the chat UI (executed when Streamlit runs this file)."""
    st.title("🍷 Wine Reviews Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about wines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = run_agent(st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    """Launch `streamlit run app.py` (the `run-app` console script)."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


if __name__ == "__main__":
    render()
