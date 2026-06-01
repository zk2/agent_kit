# Retrieval

retrieval uses hybrid search that combines pgvector vector similarity with postgres full text
search. the two result sets are fused with reciprocal rank fusion. when retrieval confidence is
low the agent returns a no answer fallback instead of guessing.
