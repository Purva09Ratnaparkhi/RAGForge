"""Quick pipeline test script."""
import sys
import requests
import json

# Force UTF-8 encoding for stdout to prevent encoding errors on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API = "http://localhost:8000"

# Test 1: Health
print("=" * 60)
print("TEST 1: Health Check")
print("=" * 60)
r = requests.get(f"{API}/health")
print(json.dumps(r.json(), indent=2))

# Test 2: Documents
print("\n" + "=" * 60)
print("TEST 2: Indexed Documents")
print("=" * 60)
r = requests.get(f"{API}/documents")
print(json.dumps(r.json(), indent=2))

# Test 3: Simple query about BERT
print("\n" + "=" * 60)
print("TEST 3: BERT Query (cross-document)")
print("=" * 60)
r = requests.post(
    f"{API}/query/sync",
    json={"query": "How does BERT pre-training work?", "history": []},
    timeout=180,
)
data = r.json()
print("Answer:", data.get("answer", "N/A")[:500])
sources = data.get("sources", [])
unique_files = set(s.get("source", "") for s in sources)
print(f"\nSource files: {unique_files}")
print(f"Total source chunks: {len(sources)}")
gr = data.get("guardrail_result", {})
print(f"\nGrounded: {gr.get('grounded')}")
print(f"Confidence: {gr.get('confidence_score')}")
print(f"Hallucination flags: {gr.get('hallucination_flags', [])}")
print(f"Warning: {gr.get('warning')}")

# Test 4: Complex query
print("\n" + "=" * 60)
print("TEST 4: Complex Query (attention mechanisms)")
print("=" * 60)
r = requests.post(
    f"{API}/query/sync",
    json={"query": "Compare multi-head attention with self-attention", "history": []},
    timeout=180,
)
data = r.json()
print("Answer:", data.get("answer", "N/A")[:500])
gr = data.get("guardrail_result", {})
print(f"\nGrounded: {gr.get('grounded')}")
print(f"Confidence: {gr.get('confidence_score')}")

# Test 5: Out of scope
print("\n" + "=" * 60)
print("TEST 5: Out-of-scope Query")
print("=" * 60)
r = requests.post(
    f"{API}/query/sync",
    json={"query": "What is the recipe for chocolate cake?", "history": []},
    timeout=180,
)
data = r.json()
print("Answer:", data.get("answer", "N/A")[:300])

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
