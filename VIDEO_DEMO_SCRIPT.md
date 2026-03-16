# ERCS Video Demo Script (10 minutes)

## Segment A: Before Running the Experiment (~5 minutes)

### 1. Project Overview (1 min)

Open the repository in the editor. Show the project structure:

```
src/ercs/           -- 6-phase simulation framework
app/dashboard.py    -- Interactive Streamlit dashboard
tests/              -- 411 tests
configs/            -- Single source of truth (default.yaml)
```

Brief verbal summary:
> "This is ERCS -- an Emergency Response Coordination Simulator. It's a discrete-event simulation that compares two coordination algorithms for emergency responders operating under degraded network infrastructure. The question is: can network-aware adaptive scheduling outperform simple proximity-based coordination when communication is intermittent?"

### 2. Tests Passing (30 sec)

Run in terminal:
```bash
pytest -q
```

Show 411 tests passing. Quick mention:
> "411 tests covering all simulation phases, with >80% code coverage."

### 3. Dashboard: Overview Tab (1 min)

Open the dashboard:
```bash
streamlit run app/dashboard.py
```

Walk through:
- Research questions (MRQ, SQ1, SQ2)
- Architecture diagram (6-phase stack)
- Experiment at a glance metrics (50 nodes, 2 algorithms, 3 connectivity levels, 180 runs)
- The two algorithms side by side

### 4. Dashboard: Parameters Tab (1 min)

- Show parameter tables with literature sources
- Highlight the network topology diagram (two-zone layout, inter-zone gap > radio range)
- Open the "Why P > 0.3?" expander -- explain the reachability filter
- Show experimental design metrics

### 5. Key Code Walkthrough (1.5 min)

Show 2-3 key code sections in the editor:

**PRoPHETv2 encounter update** (`src/ercs/communication/prophet.py`, `update_encounter` method):
> "This is the core of the delay-tolerant networking protocol. When two nodes meet, their delivery predictability updates based on time since last encounter -- recent encounters give stronger predictions."

**Adaptive scoring** (`src/ercs/coordination/algorithms.py`, `_score_candidate` method):
> "The adaptive algorithm scores responders using four factors: delivery predictability, encounter recency, physical proximity, and workload. The proximity weight is dominant at 0.6, but network awareness breaks ties."

**Deterministic encounter processing** (`src/ercs/simulation/engine.py`, `_handle_node_encounters` method):
> "We sort link tuples before processing to ensure deterministic order. Python sets iterate in hash-seed-dependent order, so without this, results would vary across processes."

### 6. Start the Experiment (30 sec)

Go back to the dashboard, Run Experiment tab:
- Toggle off "Quick test mode" for the full 30 runs/config
- Click "Start Experiment"
- Show the live progress bar starting
- Note: "This will run 180 simulations -- we'll come back when it's done."

---

## Segment B: After the Experiment Completes (~5 minutes)

### 7. Results: Run Experiment Tab (30 sec)

- Show completion message and headline metrics
- Point out the Adaptive vs Baseline delivery time difference
- Show the summary statistics table

### 8. Visualizations Tab (1.5 min)

Walk through each visualization:
- **Primary metric (avg_delivery_time grouped bars)**: "This is the main result -- the MRQ answer. Lower bars for Adaptive mean faster coordination."
- **Box plots**: "These show the distributions. Look for separation between algorithm boxes at each connectivity level."
- **Heatmap**: "Quick visual summary of all conditions."
- **Degradation lines**: "This answers SQ1 -- how does each algorithm degrade as connectivity drops? The shaded regions are 95% confidence intervals."

### 9. Network Diagnostics Tab (1.5 min)

This is the most visually impressive section:
- **Predictability graph**: "Each edge represents a delivery probability. Brighter/thicker = higher P-value. The Adaptive algorithm uses these for responder selection."
- **Predictability heatmap**: "This is the coordination-to-mobile matrix. Only cells above 0.3 pass the Adaptive algorithm's eligibility filter."
- **Predictability evolution**: "The sawtooth pattern -- encounters build predictability, aging decays it. This is PRoPHETv2 in action."
- **Message journey**: "Here's an actual message traversing the network hop by hop. You can see it moving from the coordination zone through relay nodes to the incident zone."

### 10. Statistical Analysis Tab (1 min)

- **t-test table**: Point out significant results (green highlights)
- **ANOVA table**: Compare eta-squared between algorithms
- **Effect sizes**: Interpret Cohen's d values

### 11. Key Findings Tab (30 sec)

- Show the MRQ answer with delivery time metrics
- Show SQ1 answer with eta-squared comparison
- Show SQ2 answer with delivery rate trade-off
- Brief closing: "The results show [describe what the data shows] -- adaptive network-aware coordination [does/does not] provide a significant advantage under intermittent connectivity."

---

## Key Code Sections for Video

| File | Class/Function | Why It's Interesting |
|------|---------------|---------------------|
| `communication/prophet.py` | `DeliveryPredictabilityMatrix.update_encounter()` | Core PRoPHETv2 time-scaled encounter update |
| `communication/prophet.py` | `DeliveryPredictabilityMatrix.update_transitivity()` | MAX-based transitivity (PRoPHETv2 innovation) |
| `coordination/algorithms.py` | `AdaptiveCoordinator._score_candidate()` | The 4-factor weighted scoring function |
| `coordination/algorithms.py` | `AdaptiveCoordinator._filter_eligible()` | P > 0.3 reachability filter |
| `simulation/engine.py` | `SimulationEngine._handle_node_encounters()` | Deterministic encounter processing (sorted sets) |
| `simulation/engine.py` | `SimulationEngine._process_event()` | Discrete-event main loop |
| `network/mobility.py` | `MobilityManager._assign_roles()` | Deterministic role assignment by index |
| `evaluation/metrics.py` | `StatisticalAnalyzer.independent_ttest()` | Welch's t-test implementation |

---

## Pre-Recording Checklist

- [ ] All 411 tests pass (`pytest -q`)
- [ ] `black --check src/ app/ tests/` reports no issues
- [ ] `ruff check src/ app/ tests/` reports no issues
- [ ] Dashboard loads without errors or warnings (`streamlit run app/dashboard.py`)
- [ ] Quick test mode completes successfully (toggle on, click Start)
- [ ] Full experiment run completes and all results tabs populate
- [ ] All visualizations render properly (no blank plots)
- [ ] No deprecation warnings in terminal
- [ ] `configs/default.yaml` matches all parameter values in the dissertation
- [ ] `TECHNICAL_REFERENCE.md` is accurate and complete
- [ ] README.md installation instructions work from scratch
- [ ] Terminal font is large enough for video recording
- [ ] Editor font is large enough for code walkthrough
- [ ] Dashboard is in wide mode (already set in page_config)
