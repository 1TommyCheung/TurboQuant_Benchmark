#!/usr/bin/env python3
"""Generate all prompt JSONL files for the generation benchmark.

Produces four files in the same directory as this script:
  prompts_short.jsonl    (50 entries, max_tokens=256)
  prompts_medium.jsonl   (50 entries, max_tokens=512)
  prompts_long.jsonl     (50 entries, max_tokens=1024)
  prompts_longctx.jsonl  (50 entries, max_tokens=1)

Run:  python generate_prompts.py
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
CONCEPTS = [
    "recursion", "polymorphism", "hash tables", "binary search",
    "gradient descent", "backpropagation", "TCP/IP", "REST APIs",
    "microservices", "event-driven architecture", "MapReduce",
    "the CAP theorem", "ACID transactions", "B-trees",
    "garbage collection", "monads", "coroutines", "virtual memory",
    "public-key cryptography", "consensus algorithms",
    "dependency injection", "the observer pattern", "currying",
    "lazy evaluation", "sharding", "bloom filters",
    "the PageRank algorithm", "transformers in deep learning",
    "reinforcement learning", "convolutional neural networks",
]

LANGUAGES = [
    "Python", "Rust", "Go", "TypeScript", "C++", "Java",
    "Kotlin", "Swift", "Ruby", "Haskell",
]

SHORT_TASKS = [
    "reverses a string",
    "checks if a number is prime",
    "computes the Fibonacci sequence",
    "finds the maximum element in a list",
    "counts the vowels in a string",
    "flattens a nested list",
    "removes duplicates from an array",
    "converts Celsius to Fahrenheit",
    "checks if a string is a palindrome",
    "sorts a list using bubble sort",
]

UNITS = [
    ("miles", "kilometers"),
    ("pounds", "kilograms"),
    ("Fahrenheit", "Celsius"),
    ("inches", "centimeters"),
    ("gallons", "liters"),
]

PHRASES = [
    "Hello, how are you?",
    "Where is the nearest train station?",
    "I would like a cup of coffee, please.",
    "What time does the meeting start?",
    "Thank you very much for your help.",
]

TARGET_LANGUAGES = ["French", "Spanish", "German", "Japanese", "Portuguese"]

TOPICS_A_B = [
    ("SQL", "NoSQL"),
    ("monoliths", "microservices"),
    ("REST", "GraphQL"),
    ("TCP", "UDP"),
    ("compiled languages", "interpreted languages"),
    ("inheritance", "composition"),
    ("threads", "async/await"),
    ("linked lists", "arrays"),
    ("Docker", "virtual machines"),
    ("relational databases", "document databases"),
]

CLASS_DESCRIPTIONS = [
    "implements a stack with push, pop, and peek",
    "represents a 2D vector with add, subtract, and dot product",
    "models a bank account with deposit, withdraw, and balance",
    "wraps a simple key-value cache with TTL expiry",
    "implements a priority queue using a heap",
    "represents a linked list with insert and delete",
    "models a deck of playing cards with shuffle and deal",
    "implements a basic event emitter with on and emit",
    "wraps a circular buffer with read and write",
    "represents a simple state machine with transitions",
]

# ---------------------------------------------------------------------------
# Fake code snippets for long prompts
# ---------------------------------------------------------------------------
CODE_TEMPLATES = [
    textwrap.dedent("""\
    def process_orders(orders, inventory):
        results = []
        for order in orders:
            item = order['item']
            qty = order['quantity']
            if item in inventory and inventory[item] >= qty:
                inventory[item] -= qty
                results.append({{'order_id': order['id'], 'status': 'fulfilled'}})
            else:
                results.append({{'order_id': order['id'], 'status': 'backorder'}})
        total = sum(1 for r in results if r['status'] == 'fulfilled')
        log_metric('fulfillment_rate', total / len(results))
        return results"""),
    textwrap.dedent("""\
    class DataPipeline:
        def __init__(self, source, transforms, sink):
            self.source = source
            self.transforms = transforms
            self.sink = sink
            self._stats = {{'processed': 0, 'errors': 0}}

        def run(self):
            for record in self.source.read():
                try:
                    for t in self.transforms:
                        record = t(record)
                    self.sink.write(record)
                    self._stats['processed'] += 1
                except Exception as e:
                    self._stats['errors'] += 1
                    logger.error(f"Failed: {{e}}")
            return self._stats"""),
    textwrap.dedent("""\
    async function fetchAndCache(url, cacheDuration) {{
        const cached = localStorage.getItem(url);
        if (cached) {{
            const {{ data, timestamp }} = JSON.parse(cached);
            if (Date.now() - timestamp < cacheDuration) return data;
        }}
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const data = await response.json();
        localStorage.setItem(url, JSON.stringify({{ data, timestamp: Date.now() }}));
        return data;
    }}"""),
    textwrap.dedent("""\
    fn merge_sort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {{
        if arr.len() <= 1 {{ return arr.to_vec(); }}
        let mid = arr.len() / 2;
        let left = merge_sort(&arr[..mid]);
        let right = merge_sort(&arr[mid..]);
        let mut result = Vec::with_capacity(arr.len());
        let (mut i, mut j) = (0, 0);
        while i < left.len() && j < right.len() {{
            if left[i] <= right[j] {{ result.push(left[i].clone()); i += 1; }}
            else {{ result.push(right[j].clone()); j += 1; }}
        }}
        result.extend_from_slice(&left[i..]);
        result.extend_from_slice(&right[j..]);
        result
    }}"""),
    textwrap.dedent("""\
    class RateLimiter:
        def __init__(self, max_requests, window_seconds):
            self.max_requests = max_requests
            self.window = window_seconds
            self._requests = {{}}

        def allow(self, client_id):
            now = time.time()
            if client_id not in self._requests:
                self._requests[client_id] = []
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if now - t < self.window
            ]
            if len(self._requests[client_id]) < self.max_requests:
                self._requests[client_id].append(now)
                return True
            return False"""),
]

CODE_REVIEW_QUESTIONS = [
    "Review this code for bugs, performance issues, and style problems. Suggest concrete improvements.",
    "Identify any security vulnerabilities in this code and explain how to fix them.",
    "Refactor this code to be more maintainable. Explain your changes.",
    "What edge cases does this code fail to handle? Write tests for them.",
    "Analyze the time and space complexity. Can you optimize it?",
]

# Fake document excerpts for long prompts
DOC_TEMPLATES = [
    (
        "The quarterly revenue report indicates a {pct}% increase in cloud services "
        "adoption across the enterprise segment. Key drivers include the migration of "
        "legacy workloads to containerized environments, expanded use of managed database "
        "services, and growing demand for real-time analytics platforms. The APAC region "
        "showed the strongest growth at {apac}%, while EMEA maintained steady expansion "
        "at {emea}%. Customer retention rates improved to {retention}%, attributed to "
        "enhanced SLA guarantees and dedicated support tiers. However, infrastructure "
        "costs rose by {cost}% due to increased GPU compute provisioning for AI/ML "
        "workloads. The engineering team recommends investing in spot instance optimization "
        "and reserved capacity planning to offset these costs in the next fiscal quarter."
    ),
    (
        "The incident post-mortem for outage INC-{inc} reveals a cascading failure "
        "originating in the authentication service. At {time} UTC, a routine certificate "
        "rotation triggered an unexpected TLS handshake failure between the auth proxy and "
        "the identity provider. The circuit breaker threshold was set too high at {threshold} "
        "failures, allowing {minutes} minutes of degraded service before activation. During "
        "this window, {pct}% of authentication requests failed, impacting approximately "
        "{users} users across {regions} regions. The retry storm from client-side logic "
        "amplified the load by {amp}x, further delaying recovery. Root cause: the certificate "
        "chain validation logic did not account for intermediate CA rotation. Remediation "
        "includes reducing circuit breaker thresholds, implementing graceful certificate "
        "rollover, and adding synthetic monitoring for TLS health checks."
    ),
]

DOC_QUESTIONS = [
    "Summarize the key findings and recommend three actionable next steps.",
    "What are the main risks identified? Prioritize them by severity.",
    "Extract all quantitative metrics and present them in a structured table.",
    "Write an executive summary suitable for a board presentation.",
    "Identify causal relationships in this document and draw conclusions.",
]

MULTI_STEP_PROBLEMS = [
    (
        "A distributed system processes messages from {n} independent queues. Each queue "
        "has a throughput of {tput} messages/second. Messages must be deduplicated using "
        "a bloom filter before being written to a database that supports {db_tput} writes/second. "
        "The bloom filter has a false positive rate of {fpr}%.\n\n"
        "Step 1: Calculate the total incoming message rate.\n"
        "Step 2: Determine the expected number of false positives per minute.\n"
        "Step 3: Assess whether the database can handle the write load.\n"
        "Step 4: Propose a scaling strategy if the database becomes the bottleneck.\n"
        "Step 5: Design a monitoring dashboard for this pipeline."
    ),
    (
        "An e-commerce platform receives {orders} orders per hour during peak traffic. "
        "Each order triggers {events} downstream events (inventory check, payment processing, "
        "shipping label generation, notification). The payment service has a {latency}ms p99 "
        "latency and a {error}% error rate. Failed payments are retried up to {retries} times "
        "with exponential backoff starting at {backoff}ms.\n\n"
        "Step 1: Model the expected payment processing time distribution.\n"
        "Step 2: Calculate the retry amplification factor.\n"
        "Step 3: Determine the maximum queue depth for the payment service.\n"
        "Step 4: Design a circuit breaker policy with appropriate thresholds.\n"
        "Step 5: Write pseudocode for the complete order processing pipeline."
    ),
]

# Seed paragraphs for long-context prompts
SEED_PARAGRAPHS = [
    (
        "The development of large language models has transformed the landscape of natural "
        "language processing. These models, trained on vast corpora of text data, demonstrate "
        "remarkable capabilities in understanding and generating human language. The transformer "
        "architecture, introduced in 2017, serves as the foundation for most modern language "
        "models. Self-attention mechanisms allow these models to capture long-range dependencies "
        "in text, enabling them to maintain coherence across extended passages."
    ),
    (
        "Quantum computing represents a paradigm shift in computational capability. Unlike "
        "classical bits that exist in states of 0 or 1, quantum bits (qubits) can exist in "
        "superposition, representing multiple states simultaneously. This property, combined "
        "with quantum entanglement, enables quantum computers to solve certain problems "
        "exponentially faster than classical computers. Applications include cryptography, "
        "drug discovery, optimization problems, and materials science simulations."
    ),
    (
        "Climate change mitigation strategies require coordinated global action across multiple "
        "sectors. The energy sector accounts for approximately 73% of global greenhouse gas "
        "emissions, making the transition to renewable energy sources a critical priority. Solar "
        "and wind power have seen dramatic cost reductions over the past decade, with solar PV "
        "costs declining by approximately 89% since 2010. However, grid storage and transmission "
        "infrastructure remain significant challenges for widespread renewable adoption."
    ),
    (
        "The human microbiome consists of trillions of microorganisms residing in and on the "
        "human body. Research has revealed that these microbial communities play crucial roles "
        "in digestion, immune function, mental health, and disease resistance. The gut microbiome "
        "alone contains approximately 1000 different bacterial species, collectively encoding "
        "100 times more genes than the human genome. Disruptions to this ecosystem, known as "
        "dysbiosis, have been linked to conditions ranging from inflammatory bowel disease to "
        "depression and obesity."
    ),
    (
        "Urban planning in the 21st century faces unprecedented challenges from rapid population "
        "growth, climate change, and technological disruption. Smart city initiatives leverage "
        "IoT sensors, data analytics, and AI to optimize traffic flow, energy consumption, and "
        "public services. However, these technologies raise significant privacy concerns and "
        "questions about digital equity. Successful urban development must balance efficiency "
        "gains with inclusive design principles that serve all residents regardless of socioeconomic "
        "status or technological literacy."
    ),
    (
        "Advances in gene editing technology, particularly CRISPR-Cas9, have opened new frontiers "
        "in medicine and agriculture. The ability to precisely modify DNA sequences offers potential "
        "treatments for genetic diseases, enhanced crop resilience, and novel approaches to "
        "antimicrobial resistance. Clinical trials for CRISPR-based therapies targeting sickle cell "
        "disease and certain cancers have shown promising results. Ethical frameworks continue to "
        "evolve as the technology matures, balancing therapeutic potential against concerns about "
        "germline editing and equitable access."
    ),
    (
        "The global supply chain has undergone significant transformation following recent "
        "disruptions. Just-in-time manufacturing principles, long considered optimal for cost "
        "efficiency, are being reevaluated in favor of more resilient approaches. Companies are "
        "diversifying supplier networks, increasing safety stock levels, and investing in digital "
        "supply chain twins for scenario planning. Nearshoring and reshoring trends reflect a "
        "broader shift toward supply chain sovereignty, though this comes with increased costs "
        "that must be weighed against the risk of future disruptions."
    ),
    (
        "Neuroscience research has made remarkable progress in understanding brain plasticity "
        "and its implications for learning and rehabilitation. The brain's ability to reorganize "
        "neural pathways in response to experience extends well beyond childhood development. "
        "Studies using functional MRI have demonstrated that targeted cognitive training can "
        "strengthen specific neural circuits, with applications in stroke recovery, age-related "
        "cognitive decline, and treatment of neurological disorders. Brain-computer interfaces "
        "represent a frontier technology leveraging these principles for direct neural communication."
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_entry(id_str: str, content: str, max_tokens: int, **extra) -> dict:
    entry = {
        "id": id_str,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    entry.update(extra)
    return entry


def write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name}: {len(entries)} entries, {path.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_short() -> list[dict]:
    entries: list[dict] = []
    idx = 0

    # "What is X?"
    for concept in random.sample(CONCEPTS, 15):
        idx += 1
        entries.append(make_entry(
            f"short_{idx:03d}",
            f"What is {concept}?",
            256,
        ))

    # "Write a {lang} function that {task}"
    pairs = [(l, t) for l in LANGUAGES for t in SHORT_TASKS]
    for lang, task in random.sample(pairs, 15):
        idx += 1
        entries.append(make_entry(
            f"short_{idx:03d}",
            f"Write a {lang} function that {task}.",
            256,
        ))

    # "Convert {number} {unit1} to {unit2}"
    for _ in range(10):
        idx += 1
        u1, u2 = random.choice(UNITS)
        n = random.randint(1, 1000)
        entries.append(make_entry(
            f"short_{idx:03d}",
            f"Convert {n} {u1} to {u2}.",
            256,
        ))

    # "Translate '{phrase}' to {language}"
    for phrase in PHRASES:
        for lang in random.sample(TARGET_LANGUAGES, 2):
            idx += 1
            entries.append(make_entry(
                f"short_{idx:03d}",
                f"Translate '{phrase}' to {lang}.",
                256,
            ))
            if idx >= 50:
                break
        if idx >= 50:
            break

    return entries[:50]


def generate_medium() -> list[dict]:
    entries: list[dict] = []
    idx = 0

    # "Explain {concept} with examples"
    for concept in random.sample(CONCEPTS, 20):
        idx += 1
        entries.append(make_entry(
            f"medium_{idx:03d}",
            f"Explain {concept} with examples. Include code if relevant.",
            512,
        ))

    # "Write a {lang} class that {desc}"
    pairs = [(l, d) for l in LANGUAGES for d in CLASS_DESCRIPTIONS]
    for lang, desc in random.sample(pairs, 15):
        idx += 1
        entries.append(make_entry(
            f"medium_{idx:03d}",
            f"Write a {lang} class that {desc}. Include docstrings and type hints where applicable.",
            512,
        ))

    # "Compare and contrast A and B"
    for a, b in random.sample(TOPICS_A_B, 10):
        idx += 1
        entries.append(make_entry(
            f"medium_{idx:03d}",
            f"Compare and contrast {a} and {b}. Give pros and cons of each approach.",
            512,
        ))

    # Pad to 50 with concept explanations
    remaining_concepts = [c for c in CONCEPTS if c not in [e["messages"][0]["content"] for e in entries]]
    for concept in random.sample(CONCEPTS, 50 - idx):
        idx += 1
        entries.append(make_entry(
            f"medium_{idx:03d}",
            f"Describe how {concept} is used in real-world software systems. Give at least two concrete examples.",
            512,
        ))
        if idx >= 50:
            break

    return entries[:50]


def generate_long() -> list[dict]:
    entries: list[dict] = []
    idx = 0

    # Code review prompts
    for i in range(len(CODE_TEMPLATES)):
        for q in random.sample(CODE_REVIEW_QUESTIONS, 2):
            idx += 1
            snippet = CODE_TEMPLATES[i]
            content = f"Here is a code snippet:\n\n```\n{snippet}\n```\n\n{q}"
            entries.append(make_entry(f"long_{idx:03d}", content, 1024))

    # Document analysis prompts
    for _ in range(20):
        idx += 1
        tmpl = random.choice(DOC_TEMPLATES)
        doc = tmpl.format(
            pct=random.randint(5, 45),
            apac=random.randint(10, 60),
            emea=random.randint(5, 30),
            retention=round(random.uniform(85, 99), 1),
            cost=random.randint(8, 35),
            inc=random.randint(1000, 9999),
            time=f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            threshold=random.randint(50, 500),
            minutes=random.randint(3, 30),
            users=random.randint(1000, 500000),
            regions=random.randint(2, 8),
            amp=round(random.uniform(2, 10), 1),
        )
        question = random.choice(DOC_QUESTIONS)
        content = f"Document:\n\n{doc}\n\n{question}"
        entries.append(make_entry(f"long_{idx:03d}", content, 1024))

    # Multi-step problems
    for _ in range(50 - idx):
        idx += 1
        tmpl = random.choice(MULTI_STEP_PROBLEMS)
        problem = tmpl.format(
            n=random.randint(3, 20),
            tput=random.randint(100, 10000),
            db_tput=random.randint(1000, 50000),
            fpr=round(random.uniform(0.01, 5.0), 2),
            orders=random.randint(500, 50000),
            events=random.randint(3, 8),
            latency=random.randint(50, 500),
            error=round(random.uniform(0.1, 5.0), 1),
            retries=random.randint(2, 5),
            backoff=random.choice([100, 200, 500, 1000]),
        )
        content = f"Solve the following multi-step problem:\n\n{problem}"
        entries.append(make_entry(f"long_{idx:03d}", content, 1024))

    return entries[:50]


def generate_longctx() -> list[dict]:
    entries: list[dict] = []
    targets = [1024, 4096, 8192, 32768, 131072]

    for target in targets:
        # ~4 chars per token approximation
        target_chars = target * 4
        for i in range(10):
            idx = targets.index(target) * 10 + i + 1
            # Build text by repeating and shuffling paragraphs
            paragraphs = list(SEED_PARAGRAPHS)
            text_parts: list[str] = []
            while len("\n\n".join(text_parts)) < target_chars:
                random.shuffle(paragraphs)
                for p in paragraphs:
                    text_parts.append(p)
                    if len("\n\n".join(text_parts)) >= target_chars:
                        break

            body = "\n\n".join(text_parts)
            # Trim to approximate target
            body = body[:target_chars]
            content = body + "\n\nSummarize the above text in one sentence."

            entries.append(make_entry(
                f"longctx_{idx:03d}",
                content,
                1,
                target_input_tokens=target,
            ))

    return entries[:50]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating benchmark prompt files...")

    short = generate_short()
    write_jsonl(OUT_DIR / "prompts_short.jsonl", short)

    medium = generate_medium()
    write_jsonl(OUT_DIR / "prompts_medium.jsonl", medium)

    long_ = generate_long()
    write_jsonl(OUT_DIR / "prompts_long.jsonl", long_)

    longctx = generate_longctx()
    write_jsonl(OUT_DIR / "prompts_longctx.jsonl", longctx)

    total = len(short) + len(medium) + len(long_) + len(longctx)
    print(f"\nDone. {total} total prompts across 4 files.")


if __name__ == "__main__":
    main()
