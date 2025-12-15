# CAPS (Cost-Aware ML Pipeline Selection) — Overview

This repository contains the source code for **CAPS**.  
**CAPS** acts as a middleware between the **generation** and **evaluation** stages of an AutoML system.

Its purpose is to select which pipelines should be evaluated using the **expected performance and expected cost**, controlled by a parameter **λ (lamda)**.

---

## How CAPS Works

### Cost & Performance Estimation

CAPS trains models to predict:
- the expected **execution cost** of a generated pipeline
- the expected **performance** of the same pipeline

### Pipeline Selection (NP-hard Problem)

Selecting the optimal subset of pipelines is NP-hard (see our paper for details).  
CAPS provides two approximation algorithms:
- **Greedy**
- **Beam Search**

---

## CAPS Integration with AutoML Systems

To integrate CAPS into an AutoML system, the following steps are required.

---

### 1. Pipeline Generation Hook

Call CAPS immediately after a batch of pipelines is generated.

```python
selected_indices, selected_pipelines = CAPS_middleware(
    sel_algo=self.sel_algo,
    lamda=self.lamda,
    selection=self.selection,
    data_id=self.data_id,
    random_state=self.random_state,
    N=len(sklearn_pipeline_list),
    timeout=timeout,
    sklearn_pipeline_list=sklearn_pipeline_list,
    predicted_scores=predicted_scores
)
```

**Returns**
- `selected_indices`: indices of selected pipelines in the original list
- `selected_pipelines`: list of selected sklearn pipelines

---

### 2. Pipeline Evaluation Hook

After executing the selected pipelines, update CAPS with their observed performance.

```python
CAPS_update(
    sel_algo=self.sel_algo,
    lamda=self.lamda,
    selection=self.selection,
    data_id=self.data_id,
    random_state=self.random_state,
    N=len(pipeline_scores),
    pipeline_scores=pipeline_scores
)
```

---

## Required Conditions

- The AutoML system must generate **multiple pipelines per iteration**.
- CAPS selects a subset of these pipelines.
- `selection ≤ N`.
- `N` must match the number of pipelines provided to CAPS.

---

## CAPS Parameterization

```python
sel_algo = "caps-greedy"       # caps-greedy, caps-beam_search, flaml-like, ratio
lamda = 0.5                   # trade-off between performance and cost
selection = 100               # number of pipelines to select
N = None                      # number of pipelines given for selection
sklearn_pipeline_list = []    # list of sklearn pipelines
predicted_scores = []         # expected performance of pipelines
data_id = "jannis"            # unique identifier for history graph and logging
```

---

## Example Integrations

CAPS is currently integrated with:
- **TPOT**
- **SMAC**

Integration code:
- TPOT: https://github.com/akontaxakis/tpot-caps.git
- SMAC: https://github.com/akontaxakis/SMAC3-caps.git

---

## Contact

**Antonios Kontaxakis**  
antonios.kontaxakis-ATNOSPAM-ulb.be
