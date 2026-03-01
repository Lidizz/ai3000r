"""
=============================================================================
GENERATIVE AI -- HANDS-ON LAB ASSIGNMENT
=============================================================================
Course: AI3000 - Artificial Intelligence
Topic:  Generative AI & Large Language Models

TIME: ~2 hours
GOAL: Learn by experimenting! Change values, observe effects, answer questions.
      This is NOT a programming exam -- it's a playground for understanding.

SETUP:
  1. Install dependencies:
     pip install numpy openai sentence-transformers scikit-learn

  2. Get a FREE Groq API key:
     - Go to https://console.groq.com
     - Sign up / log in
     - Go to API Keys and create a new key
     - Paste it below (line marked YOUR_API_KEY)

  3. Run this file:  python genai_assignment.py

INSTRUCTIONS:
  - Look for TODO markers -- these are your tasks
  - Look for QUESTION markers -- write your answers as comments
  - Each task is independent. If one fails, the rest still work.
  - Experiment! Change numbers, add text, see what happens.

SUBMISSION:
  - Submit this .py file with your changes and answers
=============================================================================
"""

import numpy as np
import os
import json

# ============================================================================
# CONFIGURATION -- Set your API key here
# ============================================================================
# Get your FREE key from https://console.groq.com
API_KEY = "YOUR_API_KEY"  # <-- Paste your Groq API key here

# Don't change these:
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"


# ============================================================================
# HELPER FUNCTION (provided -- do NOT modify)
# ============================================================================
def ask_llm(prompt, system_message="You are a helpful assistant.",
            temperature=1.0, max_tokens=300):
    """
    Send a prompt to the Groq API and return the response text.
    Uses the OpenAI-compatible SDK.
    """
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ##########################################################################
#
#   TASK 1: SAMPLING STRATEGIES  (~20 minutes)
#   How does an LLM decide which word to pick next?
#
# ##########################################################################
print("\n" + "=" * 70)
print("TASK 1: SAMPLING STRATEGIES")
print("=" * 70)

# --- Provided: A simulated LLM prediction ---
# Imagine the model read "The student opened the ___" and now predicts
# probabilities for the next word. These are fake scores (logits).

words  = ["book", "door", "laptop", "window", "fridge", "umbrella", "piano", "volcano"]
logits = np.array([5.0,   4.2,   3.0,     1.5,     0.5,     0.2,       -0.5,   -2.0])


def softmax(logits, temperature=1.0):
    """Convert raw scores to probabilities. Temperature controls randomness."""
    scaled = logits / temperature
    exp_vals = np.exp(scaled - np.max(scaled))
    return exp_vals / exp_vals.sum()


def top_k_filter(probs, k):
    """Keep only the top-k most likely words, zero out the rest."""
    filtered = probs.copy()
    cutoff_indices = np.argsort(probs)[:-k]
    filtered[cutoff_indices] = 0
    return filtered / filtered.sum()


def top_p_filter(probs, p):
    """Keep the smallest set of words whose probabilities sum to >= p."""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p) + 1
    filtered = np.zeros_like(probs)
    filtered[sorted_indices[:cutoff]] = probs[sorted_indices[:cutoff]]
    return filtered / filtered.sum()


# --- Show the base probabilities ---
base_probs = softmax(logits)
print("\nThe model read: 'The student opened the ___'\n")
print("Base probabilities (temperature=1.0):")
for w, p in zip(words, base_probs):
    bar = "#" * int(p * 40)
    print(f"  {w:>10s}  {p:.1%}  {bar}")

# -------------------------------------------------------------------------
# TODO 1a: Temperature experiment
# -------------------------------------------------------------------------
# Change the temperature values below and observe how the probability
# distribution changes. Try at least 4 different values.

temperatures_to_try = [0.1, 0.5, 1.0, 2.0]  # <-- Feel free to modify!

print("\n--- Temperature Experiment ---")
for temp in temperatures_to_try:
    temp_probs = softmax(logits, temperature=temp)
    top_word = words[np.argmax(temp_probs)]
    top_prob = temp_probs.max()
    print(f"  temp={temp:<5}  ->  top word: '{top_word}' at {top_prob:.1%}")

# QUESTION 1a: What happens to the probability of the top word when you
# use a very low temperature (e.g., 0.01)? What about a very high one (e.g., 10)?
# Write your answer here:
# ANSWER:


# -------------------------------------------------------------------------
# TODO 1b: Top-k experiment
# -------------------------------------------------------------------------
# Print the allowed words for k = 1, 3, and 5.
# The first one is done for you.

print("\n--- Top-k Experiment ---")
for k in [1, 3, 5]:
    topk_probs = top_k_filter(base_probs, k)
    allowed = [f"{w}" for w, p in zip(words, topk_probs) if p > 0]
    print(f"  k={k}  ->  allowed: {', '.join(allowed)}")

# TODO: Now try k = 8. What do you notice?
# Write the code here:


# QUESTION 1b: If you were building a chatbot for a bank (serious answers needed),
# would you use a low or high temperature? What about for a creative story writer?
# ANSWER:


# -------------------------------------------------------------------------
# TODO 1c: Top-p experiment
# -------------------------------------------------------------------------
# Top-p is adaptive: it includes as many words as needed to reach p% probability.
# Run this and observe.

print("\n--- Top-p Experiment ---")
for p_val in [0.5, 0.8, 0.95, 1.0]:
    topp_probs = top_p_filter(base_probs, p_val)
    n_allowed = sum(1 for p in topp_probs if p > 0)
    print(f"  p={p_val}  ->  {n_allowed} words included")

# QUESTION 1c: What is the difference between top-k and top-p?
# When would top-p be better than top-k?
# ANSWER:


# ##########################################################################
#
#   TASK 2: PROMPT ENGINEERING  (~25 minutes)
#   The SAME model gives very different answers depending on HOW you ask.
#
# ##########################################################################
print("\n" + "=" * 70)
print("TASK 2: PROMPT ENGINEERING")
print("=" * 70)

if API_KEY == "YOUR_API_KEY":
    print("\n  [!] Skipping Task 2 -- set your API key at the top of this file!\n")
else:

    # --- 2a: Zero-Shot (provided) ---
    print("\n--- 2a: Zero-Shot Prompting ---")
    zero_shot = """Classify this review as POSITIVE, NEGATIVE, or NEUTRAL.
    
Review: "The battery life is great but the camera quality is mediocre."

Classification:"""

    result = ask_llm(zero_shot, temperature=0.3)
    print(f"  Prompt: (zero-shot classification)")
    print(f"  Response: {result.strip()}")

    # -------------------------------------------------------------------------
    # TODO 2b: Few-Shot Prompting
    # -------------------------------------------------------------------------
    # Improve the classification by giving the model 2-3 examples first.
    # Fill in the prompt below:

    print("\n--- 2b: Few-Shot Prompting ---")
    few_shot = """Classify each review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "Absolutely love this product, works perfectly!"
Classification: POSITIVE

Review: "Broke after one week. Total waste of money."
Classification: NEGATIVE

Review: "It's fine, does what it says."
Classification: NEUTRAL

Review: "The battery life is great but the camera quality is mediocre."
Classification:"""

    # TODO: The few-shot prompt above is complete. Now try it:
    # Uncomment the lines below to run it.

    # result = ask_llm(few_shot, temperature=0.3)
    # print(f"  Response: {result.strip()}")

    # QUESTION 2b: Did the few-shot version give a different answer than zero-shot?
    # Which one seems more reliable? Why?
    # ANSWER:

    # -------------------------------------------------------------------------
    # TODO 2c: Chain-of-Thought Prompting
    # -------------------------------------------------------------------------
    # Ask the model to solve a reasoning problem. Compare with and without
    # "think step by step".

    print("\n--- 2c: Chain-of-Thought ---")

    # Version A: Direct question (no chain-of-thought)
    direct_prompt = """A farmer has 3 fields. Each field has 4 rows. Each row has 8 tomato 
plants. Each plant produces 5 tomatoes. 20% of all tomatoes are rotten.
How many good tomatoes are there?"""

    result_direct = ask_llm(direct_prompt, temperature=0.2)
    print(f"  Direct answer: {result_direct.strip()[:200]}")

    # TODO: Version B -- add "Think step by step" to the same question.
    # Write the modified prompt here:

    # cot_prompt = """..."""

    # result_cot = ask_llm(cot_prompt, temperature=0.2)
    # print(f"  Chain-of-thought answer: {result_cot.strip()[:500]}")

    # QUESTION 2c: Did the chain-of-thought version get the right answer?
    # (Correct answer: 384 good tomatoes). Did the direct version?
    # ANSWER:

    # -------------------------------------------------------------------------
    # TODO 2d: Role-Based Prompting
    # -------------------------------------------------------------------------
    # Ask the SAME question but with different roles. See how the answer changes.

    print("\n--- 2d: Role-Based Prompting ---")
    question = "Explain what an API is."

    roles = [
        "You are a kindergarten teacher. Explain things using simple words and fun analogies.",
        "You are a senior software engineer. Be technical and precise.",
    ]

    for role in roles:
        result = ask_llm(question, system_message=role, temperature=0.7, max_tokens=150)
        print(f"\n  Role: {role[:60]}...")
        print(f"  Answer: {result.strip()[:200]}...")

    # TODO: Add a THIRD role of your own choosing below and run it.
    # Examples: a poet, a pirate, a sports commentator, a lawyer...

    # my_role = "You are a ..."
    # result = ask_llm(question, system_message=my_role, temperature=0.7, max_tokens=150)
    # print(f"\n  Role: {my_role[:60]}...")
    # print(f"  Answer: {result.strip()[:200]}...")

    # QUESTION 2d: How much did the role change the answer? Was the core
    # information the same, or did the role change the content too?
    # ANSWER:


# ##########################################################################
#
#   TASK 3: LLM API PARAMETERS  (~20 minutes)
#   Same model, different settings = very different behavior.
#
# ##########################################################################
print("\n" + "=" * 70)
print("TASK 3: LLM API PARAMETERS")
print("=" * 70)

if API_KEY == "YOUR_API_KEY":
    print("\n  [!] Skipping Task 3 -- set your API key at the top of this file!\n")
else:

    # -------------------------------------------------------------------------
    # TODO 3a: Temperature experiment with a real LLM
    # -------------------------------------------------------------------------
    # Ask the SAME question with different temperatures. Run this TWICE to
    # see if the answers change between runs.

    print("\n--- 3a: Temperature Experiment ---")
    prompt = "Write one sentence about the future of AI."

    for temp in [0.0, 0.7, 1.5]:
        result = ask_llm(prompt, temperature=temp, max_tokens=80)
        print(f"\n  temp={temp}:")
        print(f"    {result.strip()}")

    # QUESTION 3a: Run this section TWICE. At temperature=0.0, do you get
    # the same answer both times? What about at temperature=1.5?
    # ANSWER:

    # -------------------------------------------------------------------------
    # TODO 3b: max_tokens experiment
    # -------------------------------------------------------------------------
    # max_tokens limits how long the response can be.
    # Try different values and see how it affects the answer.

    print("\n--- 3b: max_tokens Experiment ---")
    prompt = "Explain how the internet works."

    for max_tok in [20, 50, 200]:
        result = ask_llm(prompt, temperature=0.5, max_tokens=max_tok)
        print(f"\n  max_tokens={max_tok} ({len(result.split())} words):")
        print(f"    {result.strip()[:300]}")

    # QUESTION 3b: What happens when max_tokens is very small (e.g., 10)?
    # Does the model still give a useful answer?
    # ANSWER:

    # -------------------------------------------------------------------------
    # TODO 3c: Structured output (JSON)
    # -------------------------------------------------------------------------
    # Ask the model to return data in JSON format. This is essential for
    # building real applications that need to parse the response.

    print("\n--- 3c: Structured JSON Output ---")

    json_prompt = """Extract information from this text and return ONLY valid JSON.

Text: "Marie Curie was born in Warsaw, Poland in 1867. She won the Nobel 
Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."

Return JSON with keys: name, birth_city, birth_country, birth_year, achievements (list).
Respond with ONLY the JSON, no other text."""

    result = ask_llm(json_prompt, temperature=0.2)
    print(f"  Raw response:\n    {result.strip()}")

    # Try to parse it as JSON
    try:
        # Strip markdown code block markers if present
        clean = result.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:-1])
        data = json.loads(clean)
        print(f"\n  Parsed successfully!")
        print(f"    Name: {data.get('name', 'N/A')}")
        print(f"    Born: {data.get('birth_city', 'N/A')}, {data.get('birth_year', 'N/A')}")
    except json.JSONDecodeError:
        print(f"\n  Could not parse as JSON -- try lowering temperature!")

    # TODO: Write your OWN JSON extraction prompt below. Pick any text
    # (a news headline, a recipe, a sports result) and extract structured data.

    # my_prompt = """..."""
    # result = ask_llm(my_prompt, temperature=0.2)
    # print(f"\n  My JSON extraction:\n    {result.strip()}")

    # QUESTION 3c: Why is getting JSON output important for building
    # applications? What could go wrong if the model returns free text?
    # ANSWER:


# ##########################################################################
#
#   TASK 4: EMBEDDINGS & SIMILARITY  (~20 minutes)
#   How do computers understand that "king" and "queen" are related?
#
# ##########################################################################
print("\n" + "=" * 70)
print("TASK 4: EMBEDDINGS & SIMILARITY")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # --- Load the embedding model (runs locally, no API key needed!) ---
    print("\nLoading embedding model (first run downloads ~90MB)...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded!\n")

    # --- Provided: Compute similarity between sentence pairs ---
    print("--- 4a: Similarity Between Sentences ---")

    pairs = [
        ("The cat sat on the mat",  "A kitten rested on the rug"),
        ("The cat sat on the mat",  "Python is a programming language"),
        ("I love pizza",            "Pizza is my favorite food"),
        ("The weather is sunny",    "It's raining outside"),
    ]

    for sent_a, sent_b in pairs:
        emb = embed_model.encode([sent_a, sent_b])
        score = cosine_similarity([emb[0]], [emb[1]])[0][0]
        print(f"  {score:.3f}  '{sent_a}'  <->  '{sent_b}'")

    # -------------------------------------------------------------------------
    # TODO 4b: Predict before running!
    # -------------------------------------------------------------------------
    # Before running this section, PREDICT: which pair will have the highest
    # similarity? Write your prediction, then check.

    print("\n--- 4b: Your Prediction Experiment ---")

    my_pairs = [
        ("The president gave a speech",   "The leader addressed the nation"),
        ("The president gave a speech",   "I ate a sandwich for lunch"),
        ("Machine learning is powerful",  "AI can solve complex problems"),
        ("The stock market crashed",      "Financial markets declined sharply"),
    ]

    # QUESTION 4b (answer BEFORE running): Which pair above has the HIGHEST
    # similarity? Which has the LOWEST? Write your prediction:
    # PREDICTION:

    for sent_a, sent_b in my_pairs:
        emb = embed_model.encode([sent_a, sent_b])
        score = cosine_similarity([emb[0]], [emb[1]])[0][0]
        print(f"  {score:.3f}  '{sent_a}'  <->  '{sent_b}'")

    # QUESTION 4b (after running): Were you right? Any surprises?
    # ANSWER:

    # -------------------------------------------------------------------------
    # TODO 4c: Build a mini semantic search engine
    # -------------------------------------------------------------------------
    # You have a "database" of FAQ answers. Given a user question, find the
    # most relevant answer using embeddings.

    print("\n--- 4c: Mini Semantic Search ---")

    # The FAQ database (feel free to add more!)
    faq_database = [
        "To reset your password, go to Settings > Security > Change Password.",
        "Our store is open Monday to Saturday, 9am to 6pm.",
        "Free shipping is available on orders over $50.",
        "You can return any product within 30 days for a full refund.",
        "Contact support at help@example.com or call 1-800-555-HELP.",
    ]

    # Pre-compute embeddings for the database
    faq_embeddings = embed_model.encode(faq_database)

    # User questions to search:
    questions = [
        "How do I change my password?",
        "When are you open?",
        "Can I get my money back?",
    ]

    for question in questions:
        q_emb = embed_model.encode([question])
        scores = cosine_similarity(q_emb, faq_embeddings)[0]
        best_idx = np.argmax(scores)
        print(f"\n  Q: \"{question}\"")
        print(f"  -> Best match ({scores[best_idx]:.2f}): \"{faq_database[best_idx]}\"")

    # TODO: Add 2 more entries to faq_database AND write 2 new questions
    # that should match them. Run again and verify it works.

    # TODO: Try a question that does NOT match anything well. What score
    # does it get? (e.g., "What is the meaning of life?")

    # my_question = "..."
    # q_emb = embed_model.encode([my_question])
    # scores = cosine_similarity(q_emb, faq_embeddings)[0]
    # best_idx = np.argmax(scores)
    # print(f"\n  Q: \"{my_question}\"")
    # print(f"  -> Best match ({scores[best_idx]:.2f}): \"{faq_database[best_idx]}\"")

    # QUESTION 4c: The search doesn't use keywords -- it uses MEANING.
    # What are the advantages of semantic search over keyword search?
    # Give an example where keyword search would fail but semantic search works.
    # ANSWER:

except ImportError:
    print("\n  [!] sentence-transformers not installed.")
    print("  Run: pip install sentence-transformers scikit-learn")


# ##########################################################################
#
#   TASK 5: MINI RAG SYSTEM  (~25 minutes)
#   Combine retrieval + LLM to answer questions from YOUR documents.
#
# ##########################################################################
print("\n" + "=" * 70)
print("TASK 5: MINI RAG SYSTEM")
print("=" * 70)

if API_KEY == "YOUR_API_KEY":
    print("\n  [!] Skipping Task 5 (API part) -- set your API key first!\n")
    SKIP_RAG_API = True
else:
    SKIP_RAG_API = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # Reuse the model if already loaded, otherwise load it
    if 'embed_model' not in dir():
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- The knowledge base ---
    # In a real system, these would be loaded from files, databases, or websites.
    knowledge_base = [
        {
            "id": "policy-001",
            "title": "Remote Work Policy",
            "content": "Employees may work remotely up to 3 days per week. "
                       "Remote days must be approved by the direct manager at least "
                       "24 hours in advance. A stable internet connection is required. "
                       "Employees must be available on Teams during core hours (10am-4pm)."
        },
        {
            "id": "policy-002",
            "title": "Annual Leave",
            "content": "Full-time employees receive 25 days of annual leave per year. "
                       "Part-time employees receive a pro-rata amount. Leave must be "
                       "requested at least 2 weeks in advance for periods over 3 days. "
                       "Unused leave can be carried over (max 5 days) to the next year."
        },
        {
            "id": "policy-003",
            "title": "Expense Reimbursement",
            "content": "Business expenses over $25 require a receipt. Submit expenses "
                       "through the Finance Portal within 30 days. Travel meals are "
                       "capped at $60/day. Hotel bookings must be pre-approved by "
                       "your department head."
        },
        {
            "id": "policy-004",
            "title": "IT Security",
            "content": "All employees must use two-factor authentication (2FA) for "
                       "company accounts. Passwords must be at least 12 characters "
                       "with a mix of letters, numbers, and symbols. Report any "
                       "security incidents to security@company.com within 1 hour."
        },
    ]

    # TODO 5a: Add at least ONE new document to the knowledge base above.
    # Ideas: parking policy, dress code, meeting room booking, etc.
    # Just add another dictionary with "id", "title", and "content" keys.

    # --- Build the retrieval engine ---
    doc_texts = [doc["content"] for doc in knowledge_base]
    doc_embeddings = embed_model.encode(doc_texts)
    print(f"\nIndexed {len(knowledge_base)} documents.\n")

    def retrieve(query, top_k=2):
        """Find the most relevant documents for a question."""
        query_emb = embed_model.encode([query])
        scores = cosine_similarity(query_emb, doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(knowledge_base[i], scores[i]) for i in top_indices]

    # --- Test retrieval ---
    print("--- 5a: Retrieval (finding relevant documents) ---")

    test_questions = [
        "How many vacation days do I get?",
        "What is the password policy?",
        "Can I expense a hotel?",
    ]

    for question in test_questions:
        results = retrieve(question, top_k=1)
        doc, score = results[0]
        print(f"\n  Q: \"{question}\"")
        print(f"  -> [{doc['id']}] {doc['title']} (score: {score:.2f})")

    # -------------------------------------------------------------------------
    # TODO 5b: Full RAG -- Retrieval + LLM Generation
    # -------------------------------------------------------------------------
    # Now we connect retrieval to the LLM. The retrieved documents become
    # the CONTEXT that the LLM uses to answer.

    if not SKIP_RAG_API:
        print("\n--- 5b: Full RAG Pipeline ---")

        def ask_with_rag(question, top_k=2):
            """Full RAG: retrieve documents, then ask the LLM to answer."""
            # Step 1: Retrieve relevant documents
            results = retrieve(question, top_k)

            # Step 2: Build context from retrieved documents
            context_parts = []
            for doc, score in results:
                context_parts.append(f"[{doc['id']} - {doc['title']}]\n{doc['content']}")
            context = "\n\n".join(context_parts)

            # Step 3: Build the RAG prompt
            rag_prompt = f"""Answer the question based ONLY on the documents below.
If the answer is not in the documents, say "I don't have information about that."
Always mention which document(s) you used.

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""

            # Step 4: Send to LLM
            answer = ask_llm(
                rag_prompt,
                system_message="You are a helpful company assistant. Be concise and cite sources.",
                temperature=0.3,
            )
            return answer, results

        # --- Demo questions ---
        rag_questions = [
            "How many days can I work from home per week?",
            "What happens to unused vacation days?",
        ]

        for question in rag_questions:
            answer, sources = ask_with_rag(question)
            print(f"\n  Q: \"{question}\"")
            print(f"  Sources: {', '.join(doc['id'] for doc, _ in sources)}")
            print(f"  Answer: {answer.strip()[:300]}")

        # TODO 5c: Ask a question that is NOT covered by the knowledge base.
        # Example: "What is the company's policy on bringing pets to the office?"
        # What does the RAG system respond?

        # result, sources = ask_with_rag("your question here")
        # print(f"\n  Q: 'your question here'")
        # print(f"  Answer: {result.strip()}")

        # QUESTION 5c: Why is it important that the system says "I don't have
        # information" instead of making up an answer? How does RAG help prevent
        # the LLM from hallucinating?
        # ANSWER:

        # TODO 5d: If you added a new document in TODO 5a, write a question
        # that should be answered by that document. Verify it works!

        # result, sources = ask_with_rag("your question about your new document")
        # print(f"\n  Answer: {result.strip()}")

except ImportError:
    print("\n  [!] sentence-transformers not installed.")
    print("  Run: pip install sentence-transformers scikit-learn")


# ##########################################################################
#
#   FINAL REFLECTION QUESTIONS
#
# ##########################################################################
print("\n" + "=" * 70)
print("FINAL REFLECTION (answer below)")
print("=" * 70)
print("""
Answer these questions based on what you learned today:

1. You're building a customer support chatbot for a hospital.
   - What temperature would you use? Why?
   - Would you use RAG? Why or why not?

2. A friend says: "I don't need prompt engineering, I just ask ChatGPT directly."
   Give two examples where proper prompting gives significantly better results.

3. What is the relationship between embeddings (Task 4) and RAG (Task 5)?
   Could RAG work without embeddings? What would you use instead?
""")

# ANSWER 1:

# ANSWER 2:

# ANSWER 3:


print("\n" + "=" * 70)
print("DONE! Save this file with your answers and submit it.")
print("=" * 70)

