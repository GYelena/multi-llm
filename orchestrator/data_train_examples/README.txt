data_train_examples/ — minimal JSONL templates for training-data pipelines

planner.example.jsonl
  One JSON object per line. Fields: id (optional), domain (optional), query, dag.
  dag must be {"nodes":[...]} matching orchestrator.planner.parse_dag_plan.

expert.example.jsonl
  One JSON object per line. Fields: id (optional), domain (optional), expert (A|B|C),
  nodeId, taskType, query, context (object), assistant (string = desired model content).

Convert to HF chat format in your training script:
  system = fixed per expert (see ROLES.md / experts.py)
  user   = json.dumps({nodeId, taskType, query, context}, ensure_ascii=False)
  assistant = row["assistant"]
