"""
CascadeDebug Pipeline Bank Generator — Phase 1.

Generates 1000 pre-computed episodes (200 clean pipelines × 5 error injections).
This is the data backbone — the environment samples from this bank at runtime,
meaning zero LLM calls during training.

Usage:
    python data/generate_pipeline_bank.py

Output:
    data/pipeline_bank.json
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
OUTPUT_PATH = Path(__file__).parent / "pipeline_bank.json"
TOTAL_CLEAN_PIPELINES = 200
ERRORS_PER_PIPELINE = 5
ROLES = ["Researcher", "Coder", "Analyst"]

# Curriculum: level → (pipeline_length, error_types)
CURRICULUM = {
    1: {"lengths": [3], "error_types": ["wrong_type", "factual_error"]},
    2: {"lengths": [4], "error_types": ["off_by_one", "logic_inversion"]},
    3: {"lengths": [5, 6], "error_types": ["cascade_dependency"]},
}

# Distribution: how many clean pipelines per curriculum level
# ~67 level 1, ~67 level 2, ~66 level 3 = 200 total
LEVEL_DISTRIBUTION = {1: 67, 2: 67, 3: 66}

# ──────────────────────────────────────────────────────────────────────────────
# 10 Domain Families × 20 Task Variations = 200 unique tasks
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    "algorithm_analysis": {
        "briefs": [
            "Analyze the time complexity of merge sort and implement it in Python.",
            "Explain quicksort's average vs worst case and implement the partition step.",
            "Describe the differences between BFS and DFS, then implement BFS on an adjacency list.",
            "Analyze the space complexity of recursive Fibonacci and implement memoized version.",
            "Compare heap sort and merge sort tradeoffs, then implement a min-heap.",
            "Explain Dijkstra's algorithm complexity and implement it for a weighted graph.",
            "Analyze binary search tree operations and implement insert and search.",
            "Describe dynamic programming for the knapsack problem and implement it.",
            "Compare linear search and binary search, implement both with benchmarking.",
            "Analyze the amortized cost of hash table operations and implement a simple hash map.",
        ],
        "researcher_templates": [
            "{algo} has {complexity} time complexity. Space complexity is {space}. {detail}",
        ],
        "coder_templates": [
            "def {func_name}({params}):\n    {docstring}\n    {body}\n    return {retval}",
        ],
        "analyst_templates": [
            "The implementation correctly handles {case}. Tested on {test_data}. Complexity verified: {complexity}.",
        ],
    },
    "data_processing": {
        "briefs": [
            "Parse a CSV dataset of sales records, compute monthly totals, and generate a summary report.",
            "Read a JSON log file, extract error entries, and produce an error frequency table.",
            "Process a tab-separated file of student grades, calculate averages, and flag failing students.",
            "Parse XML weather data, compute weekly averages, and format output as JSON.",
            "Read a large text file line by line, count word frequencies, and output top-20 words.",
            "Process a dataset of timestamps, detect gaps larger than 1 hour, and report anomalies.",
            "Parse a nested JSON API response, flatten it, and export as CSV.",
            "Read multiple CSV files, merge on a common key, and compute aggregated statistics.",
            "Process a log file to extract request durations and compute P50/P95/P99 latencies.",
            "Parse a HTML table of financial data, clean currency formats, and compute totals.",
        ],
        "researcher_templates": [
            "The data format is {fmt}. Key columns are: {cols}. Processing approach: {approach}. Expected output shape: {shape}.",
        ],
        "coder_templates": [
            "import {lib}\n\ndef process_data(filepath):\n    {read_logic}\n    {transform}\n    return {result}",
        ],
        "analyst_templates": [
            "Data processed successfully. {n_rows} rows handled. Key finding: {finding}. Output saved to {dest}.",
        ],
    },
    "api_design": {
        "briefs": [
            "Design a REST endpoint for user authentication with JWT token generation and test it.",
            "Create a CRUD API for a product catalog with pagination and filtering.",
            "Design an API endpoint that accepts file uploads, validates format, and returns metadata.",
            "Build a rate-limited API endpoint with proper error codes and retry-after headers.",
            "Design a webhook receiver that validates signatures and processes event payloads.",
            "Create an API endpoint for search with fuzzy matching, sorting, and result ranking.",
            "Design a batch processing API that accepts bulk operations and returns a job status.",
            "Build an API versioning strategy with backward-compatible schema evolution.",
            "Create an endpoint for real-time notifications using server-sent events (SSE).",
            "Design an API for multi-tenant access with proper isolation and role-based permissions.",
        ],
        "researcher_templates": [
            "Endpoint: {method} {path}. Auth: {auth}. Request format: {req}. Response: {resp}. Status codes: {codes}.",
        ],
        "coder_templates": [
            "from fastapi import FastAPI, {imports}\n\napp = FastAPI()\n\n@app.{method}(\"{path}\")\ndef {handler}({params}):\n    {body}\n    return {response}",
        ],
        "analyst_templates": [
            "API endpoint tested with {n_cases} cases. Status codes verified: {codes}. Edge cases covered: {edges}. Performance: {perf}.",
        ],
    },
    "database_query": {
        "briefs": [
            "Write SQL queries to find the top 10 customers by total order value and visualize results.",
            "Design a database schema for a blog platform and write queries for common operations.",
            "Write an optimized query to find all users who made purchases in consecutive months.",
            "Create an SQL query with window functions to calculate running averages of daily sales.",
            "Design a query to detect duplicate records across multiple tables and report them.",
            "Write a recursive CTE query to traverse a hierarchical employee-manager structure.",
            "Optimize a slow query with proper indexing strategy and explain the execution plan.",
            "Write queries to compute A/B test metrics with statistical significance from raw event data.",
            "Create a materialized view for a dashboard that aggregates monthly KPIs.",
            "Write a query to implement a leaderboard with rank, percentile, and tie-breaking logic.",
        ],
        "researcher_templates": [
            "Schema: {tables}. Key relationships: {joins}. Query strategy: {strategy}. Expected result: {result}.",
        ],
        "coder_templates": [
            "-- {comment}\nSELECT {columns}\nFROM {table}\n{joins}\nWHERE {conditions}\n{group_order}\nLIMIT {limit};",
        ],
        "analyst_templates": [
            "Query returns {n_rows} rows. Execution plan cost: {cost}. Index usage: {idx}. Result validated: {validation}.",
        ],
    },
    "ml_pipeline": {
        "briefs": [
            "Train a logistic regression classifier on the iris dataset and evaluate with cross-validation.",
            "Build a text classification pipeline using TF-IDF features and a naive Bayes classifier.",
            "Implement a train/test split, train a random forest, and plot feature importance.",
            "Preprocess a dataset with missing values, encode categoricals, and fit a gradient boosting model.",
            "Train a k-means clustering model, determine optimal k using the elbow method, and visualize clusters.",
            "Build a regression model to predict house prices, evaluate with RMSE and R-squared.",
            "Implement a cross-validated hyperparameter search for an SVM classifier.",
            "Build a simple recommendation system using collaborative filtering on a ratings matrix.",
            "Train a decision tree, extract rules, and evaluate interpretability vs accuracy tradeoff.",
            "Implement PCA for dimensionality reduction, then train a classifier on reduced features.",
        ],
        "researcher_templates": [
            "Dataset: {dataset}. Features: {features}. Target: {target}. Approach: {approach}. Baseline: {baseline}.",
        ],
        "coder_templates": [
            "from sklearn.{module} import {cls}\nimport numpy as np\n\ndef train_model(X, y):\n    model = {cls}({params})\n    model.fit(X, y)\n    return model",
        ],
        "analyst_templates": [
            "Model accuracy: {acc}. F1-score: {f1}. Cross-validation mean: {cv}. Confusion matrix shows {confusion}. {recommendation}.",
        ],
    },
    "text_processing": {
        "briefs": [
            "Build a text tokenizer that handles punctuation, contractions, and count word frequencies.",
            "Implement a regex-based email validator with domain extraction and batch processing.",
            "Build a Markdown to HTML converter supporting headers, bold, italic, and links.",
            "Implement a text diff tool that shows additions, deletions, and modifications between two files.",
            "Build a simple spell checker using edit distance and a dictionary lookup.",
            "Create a text summarizer that extracts key sentences based on TF-IDF scoring.",
            "Implement a log parser that extracts structured fields from unstructured log lines.",
            "Build a citation formatter that converts between APA, MLA, and Chicago styles.",
            "Implement a templating engine that replaces variables and handles conditional blocks.",
            "Build a code comment extractor that parses Python files and generates documentation.",
        ],
        "researcher_templates": [
            "Text processing approach: {approach}. Input format: {input_fmt}. Expected patterns: {patterns}. Edge cases: {edges}.",
        ],
        "coder_templates": [
            "import re\n\ndef process_text(text):\n    {preprocessing}\n    {core_logic}\n    return {result}",
        ],
        "analyst_templates": [
            "Processed {n_docs} documents. Accuracy on test set: {acc}. Edge cases handled: {edges}. Performance: {perf}.",
        ],
    },
    "math_statistics": {
        "briefs": [
            "Compute confidence intervals for A/B test results and determine statistical significance.",
            "Implement a Monte Carlo simulation to estimate pi and analyze convergence rate.",
            "Calculate descriptive statistics (mean, median, std, skewness) for a dataset and visualize distributions.",
            "Implement a chi-squared test for independence on a contingency table.",
            "Build a linear regression from scratch using gradient descent and compare with closed-form solution.",
            "Implement Bayesian updating for a coin flip experiment and plot posterior distributions.",
            "Calculate correlation matrices for multivariate data and identify collinear features.",
            "Implement a bootstrap resampling method to estimate confidence intervals without assumptions.",
            "Build a moving average filter and compare SMA vs EMA on noisy time series data.",
            "Implement a hypothesis test framework supporting t-test, z-test, and Mann-Whitney U test.",
        ],
        "researcher_templates": [
            "Statistical method: {method}. Assumptions: {assumptions}. Parameters: {params}. Expected outcome: {outcome}.",
        ],
        "coder_templates": [
            "import numpy as np\nfrom scipy import stats\n\ndef compute_{func}(data):\n    {calculations}\n    return {result}",
        ],
        "analyst_templates": [
            "Results: {metric} = {value}. P-value: {pval}. Confidence interval: [{ci_low}, {ci_high}]. Conclusion: {conclusion}.",
        ],
    },
    "web_scraping": {
        "briefs": [
            "Extract product data from an e-commerce HTML page and normalize prices to USD.",
            "Scrape a news website's headline feed and categorize articles by topic.",
            "Extract structured data from a Wikipedia infobox and convert to a flat dictionary.",
            "Parse a restaurant menu from HTML, extract items with prices, and compute averages by category.",
            "Scrape job listings, extract salary ranges, and compute statistics by job title.",
            "Extract table data from a government statistics page and convert to clean CSV.",
            "Parse a forum thread to extract post timestamps, authors, and reply chains.",
            "Scrape weather forecast data, parse temperature ranges, and plot weekly trends.",
            "Extract event information from a calendar page and output structured iCal format.",
            "Parse a product review page, extract ratings and sentiments, and compute aggregate scores.",
        ],
        "researcher_templates": [
            "Target page structure: {structure}. Data fields: {fields}. Extraction strategy: {strategy}. Anti-scraping measures: {measures}.",
        ],
        "coder_templates": [
            "from bs4 import BeautifulSoup\nimport requests\n\ndef scrape_data(url):\n    {fetch_logic}\n    soup = BeautifulSoup(html, 'html.parser')\n    {extraction}\n    return {result}",
        ],
        "analyst_templates": [
            "Scraped {n_items} items. Data quality: {quality}. Missing fields: {missing}. Normalized {n_normalized} entries.",
        ],
    },
    "file_system": {
        "briefs": [
            "Recursively scan directories and report file size statistics by extension.",
            "Implement a file deduplication tool that finds and reports duplicate files by hash.",
            "Build a backup utility that copies modified files and maintains a change log.",
            "Create a directory tree visualization tool with size annotations.",
            "Implement a file watcher that detects changes and triggers callbacks.",
            "Build a disk usage analyzer that identifies the largest files and directories.",
            "Create a file organizer that sorts files into folders by type, date, or name pattern.",
            "Implement a log rotation utility that archives old logs and manages retention.",
            "Build a file search tool supporting glob patterns, regex, and content search.",
            "Create a file integrity checker that computes and verifies checksums.",
        ],
        "researcher_templates": [
            "File system approach: {approach}. Target: {target}. Key operations: {ops}. Performance considerations: {perf}.",
        ],
        "coder_templates": [
            "import os\nfrom pathlib import Path\n\ndef scan_files(root_dir):\n    {walk_logic}\n    {processing}\n    return {result}",
        ],
        "analyst_templates": [
            "Scanned {n_files} files across {n_dirs} directories. Total size: {total_size}. Findings: {findings}.",
        ],
    },
    "security": {
        "briefs": [
            "Validate user input, sanitize for SQL injection, and log suspicious attempts.",
            "Implement a password strength checker with entropy calculation and common password detection.",
            "Build an input validation library for email, URL, phone number, and credit card formats.",
            "Implement rate limiting with token bucket algorithm and log blocked requests.",
            "Create a simple encryption utility using symmetric keys for file encryption/decryption.",
            "Build a CSRF token generator and validator for web form protection.",
            "Implement an access control list (ACL) system with role-based permission checking.",
            "Create an audit logger that records user actions with tamper-evident checksums.",
            "Build a URL sanitizer that detects and prevents open redirect vulnerabilities.",
            "Implement a secure session manager with expiry, rotation, and revocation.",
        ],
        "researcher_templates": [
            "Security domain: {domain}. Threat model: {threats}. Mitigation strategy: {strategy}. Standards: {standards}.",
        ],
        "coder_templates": [
            "import hashlib\nimport re\n\ndef validate_input(user_input):\n    {sanitization}\n    {validation}\n    return {result}",
        ],
        "analyst_templates": [
            "Tested {n_cases} input cases. Blocked {n_blocked} malicious inputs. False positive rate: {fpr}. Coverage: {coverage}.",
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Step output generators — produce realistic content per role
# ──────────────────────────────────────────────────────────────────────────────

# Content banks for filling templates with variety
ALGO_DETAILS = [
    ("merge sort", "O(n log n)", "O(n)", "Uses divide-and-conquer approach splitting array in half recursively."),
    ("quicksort", "O(n log n) average", "O(log n)", "Pivot selection is critical — median of three is recommended."),
    ("binary search", "O(log n)", "O(1)", "Requires sorted input. Halves search space each iteration."),
    ("BFS", "O(V + E)", "O(V)", "Uses a queue. Guarantees shortest path in unweighted graphs."),
    ("DFS", "O(V + E)", "O(V)", "Uses a stack or recursion. Good for topological sorting."),
    ("Dijkstra's algorithm", "O((V + E) log V)", "O(V)", "Greedy approach. Doesn't work with negative edge weights."),
    ("hash table lookup", "O(1) amortized", "O(n)", "Requires good hash function to minimize collisions."),
    ("heap sort", "O(n log n)", "O(1)", "In-place but not stable. Uses a max-heap for ascending sort."),
    ("dynamic programming", "O(nW) for knapsack", "O(nW)", "Optimal substructure and overlapping subproblems are key properties."),
    ("insertion sort", "O(n^2)", "O(1)", "Efficient for small or nearly sorted arrays. Stable sort."),
]

NUMERIC_VALUES = {
    "accuracy": ["0.87", "0.92", "0.78", "0.95", "0.83", "0.91", "0.88", "0.94", "0.86", "0.79"],
    "f1": ["0.85", "0.90", "0.76", "0.93", "0.81", "0.89", "0.87", "0.92", "0.84", "0.77"],
    "p_value": ["0.023", "0.001", "0.045", "0.0001", "0.032", "0.012", "0.038", "0.005", "0.041", "0.008"],
    "n_rows": ["1,247", "5,832", "10,000", "42,891", "987", "3,456", "25,000", "8,192", "15,643", "2,048"],
    "n_items": ["156", "2,340", "891", "4,567", "312", "1,893", "567", "3,210", "734", "5,001"],
}


def _role_for_step(step_idx: int, pipeline_length: int) -> str:
    """
    Map a step index (0-based) to a role.
    Pattern: R, C, A, R, C, A… (cyclic)
    """
    return ROLES[step_idx % len(ROLES)]


def _generate_step_output(domain_key: str, role: str, step_idx: int, brief: str, rng: random.Random) -> str:
    """Generate a realistic output string for a given role and domain."""
    domain = DOMAINS[domain_key]

    if role == "Researcher":
        if domain_key == "algorithm_analysis":
            algo = rng.choice(ALGO_DETAILS)
            return f"{algo[0]} has {algo[1]} time complexity. Space complexity is {algo[2]}. {algo[3]}"
        elif domain_key == "data_processing":
            fmt = rng.choice(["CSV", "JSON", "TSV", "XML"])
            cols = rng.choice(["id, timestamp, value", "user_id, action, duration", "date, amount, category"])
            return f"The data format is {fmt}. Key columns are: {cols}. Processing approach: stream line-by-line for memory efficiency. Expected output shape: aggregated summary table."
        elif domain_key == "api_design":
            method = rng.choice(["POST", "GET", "PUT", "DELETE"])
            path = rng.choice(["/api/v1/users", "/api/v1/products", "/api/v1/auth/login", "/api/v1/search"])
            return f"Endpoint: {method} {path}. Auth: Bearer JWT. Request format: JSON body. Response: 200 with data + pagination. Status codes: 200, 400, 401, 404, 500."
        elif domain_key == "database_query":
            tables = rng.choice(["users, orders, products", "employees, departments, salaries", "posts, comments, users"])
            return f"Schema: {tables}. Key relationships: foreign key joins. Query strategy: use window functions for ranking. Expected result: top-N aggregated rows."
        elif domain_key == "ml_pipeline":
            dataset = rng.choice(["iris", "titanic", "wine", "breast_cancer", "digits"])
            return f"Dataset: {dataset}. Features: numerical + categorical. Target: classification label. Approach: train/test split + cross-validation. Baseline: majority class accuracy."
        elif domain_key == "text_processing":
            return f"Text processing approach: regex + tokenization. Input format: plain text UTF-8. Expected patterns: words, punctuation, whitespace. Edge cases: contractions, hyphenated words, Unicode."
        elif domain_key == "math_statistics":
            method = rng.choice(["t-test", "chi-squared", "bootstrap", "Monte Carlo", "Bayesian updating"])
            return f"Statistical method: {method}. Assumptions: normal distribution of residuals. Parameters: alpha=0.05, n_samples=1000. Expected outcome: confidence interval with interpretation."
        elif domain_key == "web_scraping":
            return f"Target page structure: HTML with nested divs. Data fields: title, price, rating. Extraction strategy: CSS selectors + BeautifulSoup. Anti-scraping measures: rate limiting, User-Agent rotation."
        elif domain_key == "file_system":
            return f"File system approach: recursive os.walk(). Target: all files in directory tree. Key operations: stat, hash, compare. Performance considerations: use generators for large trees."
        elif domain_key == "security":
            domain_sec = rng.choice(["SQL injection", "XSS", "CSRF", "input validation"])
            return f"Security domain: {domain_sec}. Threat model: untrusted user input. Mitigation strategy: parameterized queries + input sanitization. Standards: OWASP Top 10."
    elif role == "Coder":
        if domain_key == "algorithm_analysis":
            algo = rng.choice(ALGO_DETAILS)
            func_name = algo[0].replace(" ", "_").replace("'s", "").lower()
            return f'def {func_name}(arr):\n    """Implement {algo[0]}."""\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = {func_name}(arr[:mid])\n    right = {func_name}(arr[mid:])\n    return merge(left, right)'
        elif domain_key == "data_processing":
            lib = rng.choice(["csv", "json", "pandas"])
            return f'import {lib}\n\ndef process_data(filepath):\n    """Read and process data file."""\n    with open(filepath, "r") as f:\n        data = {lib}.load(f)\n    results = {{}}\n    for row in data:\n        key = row["category"]\n        results[key] = results.get(key, 0) + row["value"]\n    return results'
        elif domain_key == "api_design":
            handler = rng.choice(["create_user", "get_products", "login", "search_items"])
            return f'from fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.post("/api/v1/{handler}")\ndef {handler}(request: dict):\n    """Handle {handler} request."""\n    if not request.get("data"):\n        raise HTTPException(status_code=400, detail="Missing data")\n    result = process(request["data"])\n    return {{"status": "success", "data": result}}'
        elif domain_key == "database_query":
            return 'SELECT u.name, SUM(o.total) as total_spent\nFROM users u\nJOIN orders o ON u.id = o.user_id\nWHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)\nGROUP BY u.id, u.name\nORDER BY total_spent DESC\nLIMIT 10;'
        elif domain_key == "ml_pipeline":
            cls = rng.choice(["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"])
            return f'from sklearn.ensemble import {cls}\nfrom sklearn.model_selection import cross_val_score\nimport numpy as np\n\ndef train_model(X, y):\n    model = {cls}(n_estimators=100, random_state=42)\n    scores = cross_val_score(model, X, y, cv=5)\n    model.fit(X, y)\n    return model, scores.mean()'
        elif domain_key == "text_processing":
            return 'import re\n\ndef tokenize(text):\n    """Tokenize text into words."""\n    text = text.lower()\n    tokens = re.findall(r"\\b\\w+\\b", text)\n    freq = {}\n    for token in tokens:\n        freq[token] = freq.get(token, 0) + 1\n    return sorted(freq.items(), key=lambda x: -x[1])'
        elif domain_key == "math_statistics":
            return 'import numpy as np\nfrom scipy import stats\n\ndef compute_confidence_interval(data, confidence=0.95):\n    """Compute confidence interval for the mean."""\n    n = len(data)\n    mean = np.mean(data)\n    se = stats.sem(data)\n    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)\n    return mean, mean - h, mean + h'
        elif domain_key == "web_scraping":
            return 'from bs4 import BeautifulSoup\nimport requests\n\ndef scrape_data(url):\n    """Scrape product data from page."""\n    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})\n    soup = BeautifulSoup(response.text, "html.parser")\n    items = []\n    for item in soup.select(".product-card"):\n        items.append({"title": item.select_one("h2").text, "price": float(item.select_one(".price").text.strip("$"))})\n    return items'
        elif domain_key == "file_system":
            return 'import os\nfrom pathlib import Path\nfrom collections import defaultdict\n\ndef scan_files(root_dir):\n    """Scan directory tree and report statistics."""\n    stats = defaultdict(lambda: {"count": 0, "total_size": 0})\n    for dirpath, _, filenames in os.walk(root_dir):\n        for fname in filenames:\n            ext = Path(fname).suffix or ".no_ext"\n            size = os.path.getsize(os.path.join(dirpath, fname))\n            stats[ext]["count"] += 1\n            stats[ext]["total_size"] += size\n    return dict(stats)'
        elif domain_key == "security":
            return 'import re\nimport hashlib\n\ndef validate_input(user_input):\n    """Sanitize and validate user input."""\n    # Remove SQL injection patterns\n    dangerous = re.compile(r"(--|;|DROP|DELETE|UPDATE|INSERT|SELECT)\\s", re.IGNORECASE)\n    if dangerous.search(user_input):\n        return {"valid": False, "reason": "Potential SQL injection detected"}\n    # Sanitize HTML\n    sanitized = re.sub(r"<[^>]+>", "", user_input)\n    return {"valid": True, "sanitized": sanitized}'
    elif role == "Analyst":
        n_rows = rng.choice(NUMERIC_VALUES["n_rows"])
        acc = rng.choice(NUMERIC_VALUES["accuracy"])
        if domain_key in ["algorithm_analysis", "ml_pipeline"]:
            return f"The implementation correctly handles edge cases (empty input, single element, duplicates). Tested on {n_rows} cases. Complexity verified: matches theoretical analysis. Accuracy: {acc}."
        elif domain_key in ["data_processing", "web_scraping", "file_system"]:
            return f"Data processed successfully. {n_rows} rows handled. Key finding: distribution is right-skewed with outliers. Output saved to results.csv."
        elif domain_key in ["api_design", "security"]:
            return f"API endpoint tested with {n_rows} cases. Status codes verified: 200, 400, 401, 404. Edge cases covered: empty body, malformed JSON, auth expiry. Performance: <50ms p95."
        elif domain_key == "database_query":
            return f"Query returns {n_rows} rows. Execution plan cost: 0.043. Index usage: primary key + composite index on (user_id, created_at). Result validated against manual count."
        elif domain_key == "text_processing":
            return f"Processed {n_rows} documents. Accuracy on test set: {acc}. Edge cases handled: Unicode, empty strings, very long inputs. Performance: 10K docs/sec."
        elif domain_key == "math_statistics":
            pval = rng.choice(NUMERIC_VALUES["p_value"])
            return f"Results: mean = 42.3, std = 5.7. P-value: {pval}. Confidence interval: [39.1, 45.5]. Conclusion: statistically significant at alpha=0.05."

    # Fallback
    return f"[{role}] Output for step {step_idx + 1}."


# ──────────────────────────────────────────────────────────────────────────────
# Error injection
# ──────────────────────────────────────────────────────────────────────────────

ERROR_INJECTORS = {
    "wrong_type": {
        "description": "Changed return type or data format to incorrect type",
        "apply": lambda output, rng: _inject_wrong_type(output, rng),
        "keywords": lambda original: ["correct type", "proper format", "expected return"],
    },
    "factual_error": {
        "description": "Introduced a clearly wrong factual claim",
        "apply": lambda output, rng: _inject_factual_error(output, rng),
        "keywords": lambda original: _extract_factual_keywords(original),
    },
    "off_by_one": {
        "description": "Introduced an off-by-one numerical error",
        "apply": lambda output, rng: _inject_off_by_one(output, rng),
        "keywords": lambda original: _extract_numeric_keywords(original),
    },
    "logic_inversion": {
        "description": "Inverted a comparison or logical condition",
        "apply": lambda output, rng: _inject_logic_inversion(output, rng),
        "keywords": lambda original: _extract_logic_keywords(original),
    },
    "cascade_dependency": {
        "description": "Changed a variable or reference that downstream steps depend on",
        "apply": lambda output, rng: _inject_cascade_dependency(output, rng),
        "keywords": lambda original: _extract_dependency_keywords(original),
    },
}

# Factual replacements bank
FACTUAL_SWAPS = [
    ("O(n log n)", "O(n^2)"),
    ("O(log n)", "O(n)"),
    ("O(1) amortized", "O(n)"),
    ("O(V + E)", "O(V * E)"),
    ("O(n)", "O(1)"),
    ("normal distribution", "uniform distribution"),
    ("statistically significant", "not statistically significant"),
    ("foreign key joins", "cross joins without filtering"),
    ("UTF-8", "ASCII-only"),
    ("parameterized queries", "string concatenation"),
    ("divide-and-conquer", "brute force"),
    ("logarithmic", "linear"),
    ("memoized", "non-memoized"),
    ("sorted input", "unsorted input"),
    ("0-based indexing", "1-based indexing"),
    ("stable sort", "unstable sort"),
    ("correctly handles", "fails to handle"),
    ("success", "failure"),
    ("true", "false"),
    ("ascending", "descending"),
]


def _inject_wrong_type(output: str, rng: random.Random) -> str:
    """Replace a data value with the wrong type representation."""
    type_swaps = [
        ("return results", "return str(results)"),
        ("return dict(", "return list("),
        ("return model", 'return "model_placeholder"'),
        ("return items", "return len(items)"),
        ("return sorted(", "return ''.join("),
        ('float(', 'str('),
        ('int(', 'str('),
    ]
    for orig, repl in rng.sample(type_swaps, len(type_swaps)):
        if orig in output:
            return output.replace(orig, repl, 1)
    # Fallback: wrap output in wrong type indicator
    return output.replace(".", ". [NOTE: returns string instead of expected dict].", 1)


def _inject_factual_error(output: str, rng: random.Random) -> str:
    """Replace a factual claim with an incorrect one."""
    shuffled = rng.sample(FACTUAL_SWAPS, len(FACTUAL_SWAPS))
    for orig, wrong in shuffled:
        if orig.lower() in output.lower():
            # Case-insensitive replacement
            idx = output.lower().find(orig.lower())
            return output[:idx] + wrong + output[idx + len(orig):]
    # Fallback: add false claim
    return output + " Note: this algorithm has O(1) time complexity for all inputs."


def _inject_off_by_one(output: str, rng: random.Random) -> str:
    """Introduce subtle off-by-one error."""
    import re
    # Find numbers and offset by 1
    numbers = list(re.finditer(r'\b(\d+)\b', output))
    if numbers:
        match = rng.choice(numbers)
        num = int(match.group())
        new_num = num + rng.choice([-1, 1])
        return output[:match.start()] + str(new_num) + output[match.end():]
    # Fallback: alter range
    range_swaps = [
        ("range(1, n+1)", "range(1, n)"),
        ("range(0, n)", "range(0, n-1)"),
        ("<= n", "< n"),
        (">= 0", "> 0"),
        ("n - 1", "n"),
        ("n + 1", "n"),
        ("len(arr) - 1", "len(arr)"),
    ]
    for orig, repl in rng.sample(range_swaps, len(range_swaps)):
        if orig in output:
            return output.replace(orig, repl, 1)
    return output.replace(".", ". (Note: boundary is inclusive, not exclusive).", 1)


def _inject_logic_inversion(output: str, rng: random.Random) -> str:
    """Invert a comparison or logical condition."""
    logic_swaps = [
        ("DESC", "ASC"),
        ("ASC", "DESC"),
        (">", "<"),
        (">=", "<="),
        ("<=", ">="),
        ("<", ">"),
        ("True", "False"),
        ("true", "false"),
        ("ascending", "descending"),
        ("descending", "ascending"),
        ("max", "min"),
        ("min", "max"),
        ("top", "bottom"),
        ("highest", "lowest"),
        ("lowest", "highest"),
        ("increase", "decrease"),
    ]
    for orig, repl in rng.sample(logic_swaps, len(logic_swaps)):
        if orig in output:
            return output.replace(orig, repl, 1)
    return output.replace("correctly", "incorrectly", 1) if "correctly" in output else output + " (result is inverted)"


def _inject_cascade_dependency(output: str, rng: random.Random) -> str:
    """Change a variable/reference name that breaks downstream dependencies."""
    dep_swaps = [
        ("results", "temp_data"),
        ("model", "classifier_v2"),
        ("data", "raw_input"),
        ("output", "intermediate"),
        ("total", "subtotal"),
        ("user_id", "account_id"),
        ("mean", "median"),
        ("accuracy", "precision"),
        ("score", "rating"),
    ]
    for orig, repl in rng.sample(dep_swaps, len(dep_swaps)):
        if orig in output and orig not in ("a", "the", "in"):
            return output.replace(orig, repl)
    return output.replace("return", "yield", 1) if "return" in output else output + " [variable_name_changed]"


def _extract_factual_keywords(original: str) -> list[str]:
    """Extract keywords that should appear in a correct fix."""
    keywords = []
    for orig, wrong in FACTUAL_SWAPS:
        if orig.lower() in original.lower():
            keywords.append(orig)
    if not keywords:
        keywords = ["correct", "accurate", "verified"]
    return keywords[:5]


def _extract_numeric_keywords(original: str) -> list[str]:
    """Extract numeric-related keywords."""
    import re
    nums = re.findall(r'\b\d+\b', original)
    keywords = nums[:3] if nums else []
    keywords.extend(["boundary", "range", "inclusive"])
    return keywords[:5]


def _extract_logic_keywords(original: str) -> list[str]:
    """Extract logic-related keywords."""
    keywords = []
    logic_terms = ["ascending", "descending", "greater", "less", "max", "min", "top", "true", "false"]
    for term in logic_terms:
        if term.lower() in original.lower():
            keywords.append(term)
    if not keywords:
        keywords = ["correct comparison", "proper ordering"]
    return keywords[:5]


def _extract_dependency_keywords(original: str) -> list[str]:
    """Extract dependency-related keywords."""
    dep_terms = ["results", "model", "data", "output", "total", "return", "score"]
    keywords = [t for t in dep_terms if t in original]
    if not keywords:
        keywords = ["consistent naming", "matching variables"]
    return keywords[:5]


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_clean_pipeline(
    pipeline_id: int,
    domain_key: str,
    brief: str,
    pipeline_length: int,
    curriculum_level: int,
    rng: random.Random,
) -> dict:
    """Generate one clean (error-free) pipeline."""
    steps = []
    for i in range(pipeline_length):
        role = _role_for_step(i, pipeline_length)
        output = _generate_step_output(domain_key, role, i, brief, rng)
        steps.append({
            "step_id": i + 1,
            "role": role,
            "output": output,
        })
    return {
        "pipeline_id_base": f"pipe_{pipeline_id:03d}",
        "curriculum_level": curriculum_level,
        "task_brief": brief,
        "domain": domain_key,
        "clean_pipeline": steps,
    }


def inject_errors(
    clean: dict,
    error_idx: int,
    rng: random.Random,
    curriculum_level: int,
) -> dict:
    """
    Generate one corrupted episode from a clean pipeline.
    Injects one error at a uniformly random step.
    """
    steps = clean["clean_pipeline"]
    pipeline_length = len(steps)
    curriculum_config = CURRICULUM[curriculum_level]

    # Uniformly random injection step
    inject_step_idx = rng.randint(0, pipeline_length - 1)
    inject_step = steps[inject_step_idx]

    # Choose error type from allowed types for this curriculum level
    error_type = rng.choice(curriculum_config["error_types"])
    injector = ERROR_INJECTORS[error_type]

    # Apply error
    original_output = inject_step["output"]
    corrupted_output = injector["apply"](original_output, rng)

    # Build corrupted pipeline (deep copy)
    corrupted_steps = []
    for s in steps:
        step_copy = dict(s)
        if s["step_id"] == inject_step["step_id"]:
            step_copy["output"] = corrupted_output
        corrupted_steps.append(step_copy)

    # Build episode
    pipeline_id = f"{clean['pipeline_id_base']}_err_{error_idx}"
    fix_keywords = injector["keywords"](original_output)

    return {
        "pipeline_id": pipeline_id,
        "curriculum_level": curriculum_level,
        "task_brief": clean["task_brief"],
        "domain": clean["domain"],
        "clean_pipeline": [dict(s) for s in steps],
        "corrupted_pipeline": corrupted_steps,
        "injected_step": inject_step["step_id"],
        "injected_role": inject_step["role"],
        "error_type": error_type,
        "error_description": f"{injector['description']} at step {inject_step['step_id']} ({inject_step['role']})",
        "original_output": original_output,
        "corrupted_output": corrupted_output,
        "expected_fix_keywords": fix_keywords,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_pipeline_bank() -> list[dict]:
    """Generate the full pipeline bank (1000 episodes)."""
    rng = random.Random(SEED)
    domain_keys = list(DOMAINS.keys())
    all_episodes = []
    pipeline_counter = 0

    for level, n_clean in LEVEL_DISTRIBUTION.items():
        config = CURRICULUM[level]
        # 10 domains, distribute evenly: n_clean // 10 per domain, remainder to first domains
        per_domain = n_clean // len(domain_keys)
        remainder = n_clean % len(domain_keys)

        for d_idx, domain_key in enumerate(domain_keys):
            n_for_domain = per_domain + (1 if d_idx < remainder else 0)
            domain_briefs = DOMAINS[domain_key]["briefs"]

            for j in range(n_for_domain):
                brief = domain_briefs[j % len(domain_briefs)]
                length = rng.choice(config["lengths"])

                clean = generate_clean_pipeline(
                    pipeline_id=pipeline_counter,
                    domain_key=domain_key,
                    brief=brief,
                    pipeline_length=length,
                    curriculum_level=level,
                    rng=rng,
                )
                pipeline_counter += 1

                # 5 error injections per clean pipeline
                for err_idx in range(ERRORS_PER_PIPELINE):
                    episode = inject_errors(clean, err_idx, rng, level)
                    all_episodes.append(episode)

    return all_episodes


def validate_bank(episodes: list[dict]) -> dict:
    """Validate the generated pipeline bank and return statistics."""
    stats = {
        "total_episodes": len(episodes),
        "unique_ids": len(set(ep["pipeline_id"] for ep in episodes)),
        "curriculum_distribution": dict(Counter(ep["curriculum_level"] for ep in episodes)),
        "injection_step_distribution": dict(Counter(ep["injected_step"] for ep in episodes)),
        "role_distribution": dict(Counter(ep["injected_role"] for ep in episodes)),
        "error_type_distribution": dict(Counter(ep["error_type"] for ep in episodes)),
        "domain_distribution": dict(Counter(ep["domain"] for ep in episodes)),
    }

    # Validation checks
    errors = []
    if stats["total_episodes"] != stats["unique_ids"]:
        errors.append(f"Duplicate IDs found: {stats['total_episodes']} episodes but {stats['unique_ids']} unique IDs")

    for ep in episodes:
        required = [
            "pipeline_id", "curriculum_level", "task_brief", "clean_pipeline",
            "corrupted_pipeline", "injected_step", "injected_role", "error_type",
            "error_description", "original_output", "corrupted_output", "expected_fix_keywords",
        ]
        missing = [f for f in required if f not in ep]
        if missing:
            errors.append(f"Episode {ep.get('pipeline_id', 'UNKNOWN')} missing fields: {missing}")
        if not ep.get("expected_fix_keywords"):
            errors.append(f"Episode {ep['pipeline_id']} has empty expected_fix_keywords")

    stats["validation_errors"] = errors
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("CascadeDebug Pipeline Bank Generator")
    print("=" * 60)

    episodes = generate_pipeline_bank()
    stats = validate_bank(episodes)

    print(f"\n📊 Total episodes: {stats['total_episodes']}")
    print(f"📊 Unique IDs: {stats['unique_ids']}")
    print(f"\n📊 Curriculum distribution:")
    for level, count in sorted(stats["curriculum_distribution"].items()):
        print(f"   Level {level}: {count} episodes")
    print(f"\n📊 Injection step distribution:")
    for step, count in sorted(stats["injection_step_distribution"].items()):
        print(f"   Step {step}: {count} episodes")
    print(f"\n📊 Role distribution:")
    for role, count in sorted(stats["role_distribution"].items()):
        print(f"   {role}: {count} episodes")
    print(f"\n📊 Error type distribution:")
    for etype, count in sorted(stats["error_type_distribution"].items()):
        print(f"   {etype}: {count} episodes")
    print(f"\n📊 Domain distribution:")
    for domain, count in sorted(stats["domain_distribution"].items()):
        print(f"   {domain}: {count} episodes")

    if stats["validation_errors"]:
        print(f"\n❌ Validation errors ({len(stats['validation_errors'])}):")
        for err in stats["validation_errors"][:10]:
            print(f"   {err}")
        sys.exit(1)
    else:
        print(f"\n✅ All validations passed!")

    # Write to file
    with open(OUTPUT_PATH, "w") as f:
        json.dump(episodes, f, indent=2)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ Wrote {len(episodes)} episodes to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    print("=" * 60)
