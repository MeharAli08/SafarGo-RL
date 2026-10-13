# SafarGo Fleet Dispatch MVP

A beginner-friendly, **simulation-only** Python project for SafarGo's 40-zone ride-hailing scenario. A Streamlit dashboard trains a small Q-learning agent, compares it with two simple policies, displays evaluation metrics, and plots the reward curve.

## Project structure

```text
SafarGo-MVP/
├── app.py                    # Streamlit user interface
├── environment.py            # Synthetic 40-zone simulation
├── agent.py                  # Tabular Q-learning agent
├── baselines.py              # Demand heuristic and stay-put policies
├── evaluation.py             # Training and policy evaluation loops
├── requirements.txt          # Python package list
├── README.md                 # Setup and deployment instructions
└── RESEARCH_AND_FORMULATION.md  # Assignment formulation and research notes
```

Each Python file has one clear job. Start the dashboard from `app.py`; the simulation, learning agent, baselines, and evaluation code are kept in their own files beside it.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open PowerShell in the `SafarGo-MVP` project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install the packages:

   ```powershell
   pip install -r requirements.txt
   ```

5. Launch the dashboard:

   ```powershell
   streamlit run app.py
   ```

Streamlit prints a local address (usually `http://localhost:8501`) to open in your browser.

## What the demo simulates

- 300 vehicles across 40 zones, with a decision every five minutes.
- A full simulation day of 288 decision steps.
- Synthetic requests with stylized time-of-day, weather, and local demand pulses.
- Q-learning state buckets for time, weather, supply imbalance, and demand; 40 destination actions plus stay.
- Three policies: Q-learning, a demand-based heuristic, and stay in place.

The UI reports completed rides, passenger wait, cancellations, zones served, an illustrative revenue proxy in **PKR**, and a relative fuel proxy. The revenue proxy uses PKR 300 per completed ride solely to show local currency formatting. It is not a real fare estimate. Fuel units are not liters or rupees.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and add the files in the project structure above. Do not upload `.venv`.
2. In Streamlit Community Cloud, create a new app connected to that repository.
3. Select `app.py` as the app entry point and deploy.
4. No secrets or live ride data are needed for this simulation.

## Safety before a real-world pilot

This MVP must remain disconnected from production dispatch. Before live use, validate the simulator and policy with historical replay and shadow-mode recommendations. Any separately approved pilot should involve at most 10% of vehicles, opt-in drivers, service and fuel limits, monitoring for long waits and cancellations, and an immediate fallback to the current dispatch policy. Do not enable online exploration with real passengers.
