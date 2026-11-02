# Project Context: Legal Contract Clause Risk Checker

## Architecture & Current State
We are building a legal-tech mini project that combines QLoRA, RAG, LangGraph, and Streamlit.
- **Phase A (Data Prep)**: Complete. `data/prepare_cuad.py` created train/val JSONL files.
- **Phase C (RAG)**: Complete. `rag/indexer.py` and `retriever.py` are built and indexed in ChromaDB/BM25.
- **Phase D (LangGraph)**: Complete. `graph.py`, `nodes.py`, `state.py` have the deterministic citation validation node (`validate_citation`).
- **Phase E (Rules)**: Complete. `rules/risk_rules.py` handles deterministic flagging.
- **Phase F (Streamlit)**: Complete. `app.py` is running locally and rendering the UI + Agent Trace beautifully.

## Hardware Constraints (CRITICAL)
- **GPU**: NVIDIA RTX 4050 (6GB VRAM).
- **System RAM**: Only ~2.0 GB available. This is a severe bottleneck for loading the LLM for inference.
- **Environment**: Python venv, PyTorch 2.6+ (CUDA 12.1), running locally on Windows.

## Tasks to Implement Now

### 1. Connect QLoRA Adapter for Inference (`langgraph_agent/nodes.py`)
- Currently, `classify_clause` uses a stub. Update it to load `models/Qwen2.5-1.5B-Instruct` in 4-bit (bitsandbytes) + the LoRA adapter from `finetune/adapters/`.
- **CRITICAL**: To avoid crashing the 2GB System RAM, use `device_map="auto"`, `low_cpu_mem_usage=True`, and ensure the model unloads properly between runs, OR use a global caching mechanism so Streamlit doesn't reload it on every button click.
- Use the model to output structured JSON: `{clause_type: str, risk_level: str, explanation: str, quoted_span: str}`.

### 2. Fix Risk Rules (`rules/risk_rules.py`)
- Update the "Termination" rule to catch phrases like `"without prior written notice"`, `"terminate at any time"`, and `"unilateral termination"`.
- Ensure the rules return the `risk_level` and `rule_name` correctly.

### 3. Create Evaluation Script (`evaluate.py`)
- Write a script that evaluates the pipeline against `data/processed/risk_qa_eval.jsonl`.
- Calculate and print:
  - Classification Accuracy (of the QLoRA model).
  - Retrieval Hit Rate (does the supporting span appear in the top-k retrieved chunks).
  - Refusal Rate (feed it a fake clause, verify `citation_valid` is False and `refusal_reason` is triggered).

### 4. Streamlit Toggle (`app.py`)
- Add a checkbox in the sidebar: `[ ] Use QLoRA Adapter`.
- Pass this state down to the LangGraph pipeline. If unchecked, use the base model (or stub); if checked, use the fine-tuned adapter.
- This is crucial for the presentation to show the "Fine-Tuning Effect".

### 5. Create Demo File (`demo_contract.txt`)
- Create a synthetic contract with 4-5 clauses, deliberately planting one High Risk (uncapped liability), one Medium Risk (unilateral termination), and one Low Risk (payment terms).

### 6. Update Documentation (`README.md`)
- Document the architecture (RAG + QLoRA + LangGraph).
- Add a section explaining why this isn't a generic RAG wrapper (focus on deterministic citation validation and clause-level retrieval).
- Add setup/run instructions.