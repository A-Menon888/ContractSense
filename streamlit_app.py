"""
Minimal Streamlit frontend for ContractSense QA

Run:
  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
import time
import os
import streamlit as st

# Ensure src is on sys.path for imports
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.provenance_qa import create_qa_engine
    from src.provenance_qa.models.context_models import ContextStrategy
except Exception as e:
    raise ImportError(
        "Could not import create_qa_engine. Ensure you're running from the ContractSense directory "
        "and that the 'src' folder exists."
    ) from e


def _map_strategy(strategy_str: str) -> "ContextStrategy":
    mapping = {
        "focused": ContextStrategy.FOCUSED,
        "comprehensive": ContextStrategy.COMPREHENSIVE,
        "hierarchical": ContextStrategy.HIERARCHICAL,
        "temporal": ContextStrategy.TEMPORAL,
        "similarity": ContextStrategy.SIMILARITY,
    }
    return mapping.get(strategy_str.lower(), ContextStrategy.FOCUSED)


def get_engine(gemini_api_key: str | None, context_strategy: str, enable_validation: bool):
    if "qa_engine" not in st.session_state:
        st.session_state.qa_engine = None

    # Recreate engine if config changed
    cfg = (gemini_api_key or "", context_strategy, enable_validation)
    if st.session_state.get("engine_cfg") != cfg or st.session_state.qa_engine is None:
        st.session_state.engine_cfg = cfg
        st.session_state.qa_engine = create_qa_engine(
            gemini_api_key=gemini_api_key,
            workspace_path=str(CURRENT_DIR),
            enable_validation=enable_validation,
            context_strategy=_map_strategy(context_strategy),
        )
    return st.session_state.qa_engine


def render_response(response):
    # Primary answer
    st.markdown("**Answer**")
    st.write(getattr(response, "answer", str(response)))

    # Metrics
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{getattr(response, 'overall_confidence', 0.0):.2f}")
    cols[1].metric("Quality", f"{getattr(response, 'answer_quality', 0.0):.2f}")
    cols[2].metric("Citations", f"{len(getattr(response, 'citations', []))}")
    cols[3].metric("Time (s)", f"{getattr(response, 'processing_time', 0.0):.2f}")

    # Citations preview
    citations = getattr(response, "citations", [])
    if citations:
        with st.expander("Citations"):
            for i, c in enumerate(citations, 1):
                title = getattr(c, "source_title", None) or getattr(c, "document_title", None) or getattr(c, "title", f"Document {i}")
                preview = getattr(c, "cited_text", None) or getattr(c, "relevant_text", None) or getattr(c, "content", "")
                st.markdown(f"- **{title}**")
                if preview:
                    st.caption(preview[:300] + ("..." if len(preview) > 300 else ""))

    # Raw metadata
    meta = getattr(response, "metadata", None)
    if meta:
        with st.expander("Response metadata"):
            st.json(meta)


def main():
    st.set_page_config(page_title="ContractSense QA", page_icon="🤖", layout="centered")
    st.title("🤖 ContractSense QA")
    st.caption("Ask questions about contracts. Answers include confidence and citations.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Gemini API Key (optional)", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        strategy = st.selectbox(
            "Context Strategy",
            ["focused", "comprehensive", "hierarchical", "temporal", "similarity"],
            index=0
        )
        enable_validation = st.checkbox("Enable Validation", value=True)
        st.divider()
        st.markdown("Upload TXT files to use as sources (CUAD folder).")

        # Simple TXT uploader to feed Module 9's CUAD text loader
        upload_files = st.file_uploader("Upload contract TXT files", type=["txt"], accept_multiple_files=True)
        if upload_files:
            dest_dir = CURRENT_DIR / "CUAD_v1" / "full_contract_txt"
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for uf in upload_files:
                try:
                    content = uf.read()
                    out_path = dest_dir / uf.name
                    with open(out_path, "wb") as f:
                        f.write(content)
                    saved += 1
                except Exception as ex:
                    st.warning(f"Failed to save {uf.name}: {ex}")
            if saved:
                st.success(f"Saved {saved} file(s) to {dest_dir}")
                st.caption("Ask again to use newly added documents.")

        # Show detected TXT count for quick verification
        det_dir = CURRENT_DIR / "CUAD_v1" / "full_contract_txt"
        if det_dir.exists():
            txt_count = len(list(det_dir.glob("*.txt")))
            st.caption(f"Detected {txt_count} TXT file(s) in CUAD_v1/full_contract_txt")

    # Initialize engine lazily
    if st.button("Initialize Engine", type="secondary"):
        try:
            engine = get_engine(api_key or None, strategy, enable_validation)
            st.success("Engine initialized")
        except Exception as e:
            st.error(f"Failed to initialize engine: {e}")

    # Question input
    question = st.text_area("Your question", placeholder="What are the termination clauses in the contract?", height=100)
    ask = st.button("Ask", type="primary", use_container_width=True)

    if ask and question.strip():
        try:
            engine = get_engine(api_key or None, strategy, enable_validation)
            start = time.time()
            response = engine.ask_question(question)
            # Ensure processing_time available
            if not hasattr(response, "processing_time") or not response.processing_time:
                response.processing_time = time.time() - start
            render_response(response)

            # Save simple history
            history = st.session_state.get("history", [])
            history.append({
                "q": question,
                "a": getattr(response, "answer", ""),
                "confidence": getattr(response, "overall_confidence", 0.0),
                "time": getattr(response, "processing_time", 0.0),
            })
            st.session_state.history = history

        except Exception as e:
            st.error(f"Error processing question: {e}")

    # Show history
    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"History ({len(history)})"):
            for i, item in enumerate(history[::-1], 1):
                st.markdown(f"{i}. **Q:** {item['q']}")
                st.caption(f"Confidence: {item['confidence']:.2f} • Time: {item['time']:.2f}s")
                st.text(item['a'][:300] + ("..." if len(item['a']) > 300 else ""))


if __name__ == "__main__":
    main()


