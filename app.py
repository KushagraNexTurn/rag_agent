import os
import streamlit as st

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

st.set_page_config(page_title="Product Specs Router", page_icon="🧭", layout="wide")
st.title("🧭 Product Specifications Router (AeroFlow vs EcoSprint)")

# --- Read secrets/env for Groq key ---
# Priority: st.secrets -> env var -> text input
groq_key_from_secrets = None
try:
    groq_key_from_secrets = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    pass

env_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.header("Configuration")
    st.caption("Provide your Groq API Key via Streamlit Secrets or here directly.")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=groq_key_from_secrets or env_key,
        type="password"
    )
    model_name = st.selectbox(
        "Groq Model",
        [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ],
        index=0,
        help="If you hit rate limits, try mixtral-8x7b-32768 or llama3-8b-8192."
    )
    chunk_size = st.slider("Chunk size (chars)", min_value=256, max_value=2048, value=1024, step=128)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.1, 0.1)
    st.caption("Lower temperature can make routing more consistent.")

    st.markdown("PDF filenames (place these next to this script or provide full paths):")
    aerofile = st.text_input("AeroFlow PDF filename", value="AeroFlow_Specification_Document.pdf")
    ecosfile = st.text_input("EcoSprint PDF filename", value="EcoSprint_Specification_Document.pdf")

st.info("Enter a question below. The router will pick the most relevant product index and answer from it.")

def build_router(groq_api_key, model_name, chunk_size, aerofile, ecosfile, temperature):
    # Configure LLM and embeddings (global defaults)
    Settings.llm = Groq(model=model_name, api_key=groq_api_key, temperature=temperature)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    splitter = SentenceSplitter(chunk_size=chunk_size)

    # Load PDFs
    aero_docs = SimpleDirectoryReader(input_files=[aerofile]).load_data()
    eco_docs = SimpleDirectoryReader(input_files=[ecosfile]).load_data()

    # Build indices
    aero_nodes = splitter.get_nodes_from_documents(aero_docs)
    ecos_nodes = splitter.get_nodes_from_documents(eco_docs)

    aero_index = VectorStoreIndex(aero_nodes)
    ecos_index = VectorStoreIndex(ecos_nodes)

    aero_qe = aero_index.as_query_engine()
    ecos_qe = ecos_index.as_query_engine()

    # Tools
    aeroflow_tool = QueryEngineTool.from_defaults(
        query_engine=aero_qe,
        name="Aeroflow specifications",
        description="Contains information about Aeroflow: Design, features, technology, maintenance, warranty."
    )
    ecosprint_tool = QueryEngineTool.from_defaults(
        query_engine=ecos_qe,
        name="EcoSprint specifications",
        description="Contains information about EcoSprint: Design, features, technology, maintenance, warranty."
    )

    # Router
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[aeroflow_tool, ecosprint_tool],
        verbose=True
    )
    return router

# Initialize router
c1, c2 = st.columns([1,1])
with c1:
    init_clicked = st.button("Initialize Router", type="primary")
with c2:
    st.caption("If you change model, chunk size, or filenames, click Initialize again.")

if init_clicked:
    if not groq_api_key:
        st.error("Please provide your Groq API Key in the sidebar or via Streamlit Secrets.")
    else:
        try:
            st.session_state.router = build_router(
                groq_api_key, model_name, chunk_size, aerofile, ecosfile, temperature
            )
            st.success("Router initialized successfully.")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

# Query UI
st.subheader("Ask a question")
default_examples = [
    "What colors are available for AeroFlow?",
    "What colors are available for EcoSprint?",
    "What are design specifications?",
]
query = st.text_input("Your question", value=default_examples[0])

ex_cols = st.columns(len(default_examples))
for i, ex in enumerate(default_examples):
    if ex_cols[i].button(ex):
        query = ex
        st.experimental_rerun()

if st.button("Run Query"):
    if "router" not in st.session_state:
        st.error("Router is not initialized. Click 'Initialize Router' in the sidebar.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Routing and querying..."):
            try:
                response = st.session_state.router.query(query)
                st.success("Done.")
                st.markdown("Response:")
                st.write(str(response))
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.info(
                    "If you see JSON parsing errors from the selector, try lowering temperature "
                    "or asking a more specific question."
                )
