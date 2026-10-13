import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from evaluation import evaluate, train

st.set_page_config(
    page_title="SafarGo | Fleet Lab",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #f4f7fb; color: #14263d; }
    [data-testid="stSidebar"] { background: #10233d; }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #1b3553; }
    [data-testid="stSidebar"] input { color: #10233d !important; }
    .block-container { padding-top: 1.6rem; max-width: 1440px; }
    .hero { background: linear-gradient(115deg, #10233d, #174c62); color: white;
        border-radius: 18px; padding: 1.6rem 2rem; margin-bottom: 1rem; }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #d7e8ef; margin: .45rem 0 0 0; }
    .eyebrow { color: #73dbc4 !important; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e2eaf1;
        padding: 1rem 1.1rem; border-radius: 14px; box-shadow: 0 4px 16px #16314b0a; }
    div[data-testid="stMetricLabel"] { color: #63758b; }
    div[data-testid="stMetricValue"] { color: #10233d; }
    .notice { background: #fff7e6; border-left: 4px solid #e3a72f; padding: .85rem 1rem;
        border-radius: 8px; color: #574517; margin: .7rem 0 1.1rem; }
    h2, h3 { color: #14263d; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚕 SafarGo")
    st.caption("Dispatch & repositioning lab")
    st.divider()
    st.subheader("Simulation setup")
    episodes = st.slider("Training episodes", 10, 150, 60, 10)
    evaluation_days = st.slider("Evaluation days", 1, 10, 3)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=7)
    run = st.button("Run simulation", type="primary", width="stretch")
    st.divider()
    st.caption("Generated demand only. This app does not connect to drivers, passengers, or live dispatch.")

st.markdown("""
<div class="hero">
  <p class="eyebrow">SafarGo · Rawalpindi & Islamabad</p>
  <h1>Fleet Dispatch Simulation</h1>
  <p>Compare a simple reinforcement-learning policy with practical baseline strategies.</p>
</div>
<div class="notice"><b>Simulation only.</b> Demand, fares, wait times and fuel are illustrative proxies. Results must not guide live drivers.</div>
""", unsafe_allow_html=True)

if run or "results" not in st.session_state:
    with st.spinner("Training the agent and evaluating the strategies..."):
        agent, rewards = train(int(episodes), int(seed))
        results = {
            "Q-learning": evaluate("rl", agent, int(evaluation_days), 500 + int(seed)),
            "Demand heuristic": evaluate("demand", episodes=int(evaluation_days), seed=500 + int(seed)),
            "Stay in place": evaluate("stay", episodes=int(evaluation_days), seed=500 + int(seed)),
        }
        st.session_state["agent"] = agent
        st.session_state["rewards"] = rewards
        st.session_state["results"] = results
        st.session_state["settings"] = (int(episodes), int(evaluation_days), int(seed))

results = st.session_state.get("results")
if not results:
    st.info("Choose simulation settings in the sidebar, then select **Run simulation**.")
    st.stop()

selected_policy = st.selectbox("Show headline results for", list(results.keys()))
selected = results[selected_policy]
metrics = st.columns(4)
metrics[0].metric("Completed rides", f"{selected['Completed rides']:,.0f}")
metrics[1].metric(
    "Average passenger wait",
    f"{selected['Average wait (min)']:.1f} min",
    delta=f"{selected['Average wait (min)'] - 7:.1f} min vs 7 min target",
    delta_color="inverse",
)
metrics[2].metric("Revenue proxy", f"PKR {selected['Revenue (PKR)']:,.0f}")
metrics[3].metric("Fuel proxy", f"{selected['Fuel proxy (units)']:,.0f} units")
st.caption(
    f"Results use {st.session_state['settings'][0]} training episodes and "
    f"average {st.session_state['settings'][1]} simulated evaluation day(s)."
)

overview, comparison, guide = st.tabs(["Overview", "Compare policies", "How this demo works"])
with overview:
    reward_column, snapshot_column = st.columns([1.4, 1])
    with reward_column:
        st.subheader("Learning progress")
        reward_values = st.session_state["rewards"]
        reward_frame = pd.DataFrame({"Episode reward": reward_values})
        reward_frame["5-episode average"] = reward_frame["Episode reward"].rolling(5, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(reward_frame.index + 1, reward_frame["Episode reward"], color="#9bb2c7", alpha=.65, linewidth=1.3)
        ax.plot(reward_frame.index + 1, reward_frame["5-episode average"], color="#078f83", linewidth=2.4)
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Total reward")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with snapshot_column:
        st.subheader("Evaluation snapshot")
        st.write(f"**Selected policy:** {selected_policy}")
        st.write(f"**Completed rides:** {selected['Completed rides']:,.0f} per simulated day")
        st.write(f"**Cancelled requests:** {selected['Cancelled requests']:,.0f}")
        st.write(f"**Zones with service:** {selected['Zones served']} of 40")
        st.caption("One simulated day contains 288 five-minute decisions. The fleet is modeled in aggregate.")

with comparison:
    st.subheader("Average results across evaluation days")
    comparison_frame = pd.DataFrame(results).T
    st.dataframe(comparison_frame, width="stretch")
    st.caption("Revenue uses a made-up PKR 300 fare per ride. Fuel units are relative; they are not liters or rupees.")
    rides_column, wait_column = st.columns(2)
    with rides_column:
        st.markdown("**Completed rides**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(comparison_frame.index, comparison_frame["Completed rides"], color="#078f83")
        ax.set_ylabel("Rides per simulated day")
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with wait_column:
        st.markdown("**Average passenger wait**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(comparison_frame.index, comparison_frame["Average wait (min)"], color="#e3a72f")
        ax.axhline(7, color="#b24b4b", linestyle="--", linewidth=1.5, label="7-minute target")
        ax.set_ylabel("Minutes per request")
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with guide:
    st.subheader("What the simulator does")
    st.markdown("""
    - Represents **300 vehicles, 40 zones**, and one decision every five minutes.
    - Trains tabular Q-learning using time, weather, supply-imbalance, and demand buckets.
    - Compares Q-learning with a demand-based heuristic and a stay-in-place policy.
    - Reports rides, passenger wait, cancellations, zone coverage, a PKR fare proxy, and a fuel proxy.
    """)
    st.subheader("Important limitations")
    st.markdown("""
    The simulator does not model road-network travel times, real traffic, trip destinations or durations, driver choice, or real local demand. A production recommendation would require historical-data validation and a stronger simulator. This small Q-table does not represent every vehicle position independently.
    """)
    st.subheader("Safe path to a pilot")
    st.markdown("""
    Validate with historical replay and shadow-mode recommendations first. Any later live pilot should be capped at **10% of vehicles**, use opt-in drivers, enforce distance/fuel and zone-service limits, monitor waits and cancellations, and retain a quick fallback to current dispatch. Keep online exploration disabled around real passengers.
    """)
