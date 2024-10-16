## Demo scripts

### Ask Anything

1. Ask any questions about the given workspace

```python 

PYTHONPATH=. python scripts/run_ask.py \
  --workspace $(pwd)/benchmark/workspaces/OpenHands/39_Drug_Response_Prediction_SVM_GDSC_ML \
  --question "What does this workspace contain?"
```

### Agent-as-a-Judge

2. Using the collected trajectories or development logs (gray-box setting)

```python
PYTHONPATH=. python scripts/run_aaaj.py \
  --developer_agent "OpenHands" \
  --setting "gray_box" \
  --planning "comprehensive (no planning)" \
  --benchmark_dir $(pwd)/benchmark
```

3. Do not have trajectories or development logs (black-box setting)

```python
PYTHONPATH=. python scripts/run_aaaj.py \
  --developer_agent "OpenHands" \
  --setting "black_box" \
  --planning "efficient (no planning)" \
  --benchmark_dir $(pwd)/benchmark
```

4. Do not have trajectories or development logs and using planning to decide the actions of Agent-as-a-Judge (black-box setting)

```python
PYTHONPATH=. python scripts/run_aaaj.py \
  --developer_agent "OpenHands" \
  --setting "gray_box" \
  --planning "planning" \
  --benchmark_dir $(pwd)/benchmark
```

### Statistics

5. Get the statistics of the projects

```python
PYTHONPATH=. python scripts/run_statistics.py \
    --benchmark_dir $(pwd)/benchmark \
    --developer_agent OpenHands
```