# Run tests from T06 in one JSON file

Put both Python and JSON files in the project root, then run:

```bash
python run_rag_tests_from_06.py
```

All complete API responses and automatic evaluations are saved in one file:

```text
rag_test_results_from_T06.json
```

The suite runs T06 through T20. A 20-second delay is used by default to reduce Groq rate-limit errors.
