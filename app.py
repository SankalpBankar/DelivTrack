import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from logic import select_orders  # reuse your select_orders function

st.set_page_config(page_title="DelivTrack— Online Delivery Tracker🚚🗺️", layout="wide")

st.title("DelivTrack — Online Delivery Tracker🚚🗺️")
st.markdown(
    "Upload `df_orders.csv` from train folder from Kaggle (or use the sample). "
    "Configure sample size and capacity, then run route optimization."
)

# Sidebar controls

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload df_orders.csv", type=["csv"])
use_local = st.sidebar.checkbox("Use local df_orders.csv if upload not provided", value=True)
sample_size = st.sidebar.slider("Sample size (for route simulation)", min_value=20, max_value=1000, value=200, step=10)
max_capacity = st.sidebar.number_input("Max capacity (units)", value=10000.0, step=100.0)
run_button = st.sidebar.button("Run Route Optimization")

# Helper functions

#@st.cache_data
def load_local_dataframe():
    return pd.read_csv("df_orders.csv")

def load_dataframe(uploaded_file, use_local):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_local:
        df = load_local_dataframe()
    else:
        df = None
    return df

def prepare_dataframe(df):
    # ensure necessary timestamp columns exist
    if "order_purchase_timestamp" in df.columns and "order_approved_at" in df.columns:
        df = df.copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
        df["order_approved_at"] = pd.to_datetime(df["order_approved_at"], errors="coerce")
        df["processing_time_min"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 60
        df = df.dropna(subset=["processing_time_min"]).reset_index(drop=True)
    else:
        st.error(
            "Required timestamp columns not found in dataset. "
            "Need 'order_purchase_timestamp' and 'order_approved_at'."
        )
        return None
    return df

def sample_dataframe(df, n):
    if n >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n, random_state=42).reset_index(drop=True)

def generate_coords(df, seed=42):
    """
    Returns:
      coords_array: numpy array shape (n,2)
      coords_map: dict order_id -> (x, y)
    Preference: use latitude/longitude columns if present; otherwise create reproducible random coords.
    """
    if "latitude" in df.columns and "longitude" in df.columns:
        coords_array = df[["latitude", "longitude"]].values.astype(float)
    elif "lat" in df.columns and "lon" in df.columns:
        coords_array = df[["lat", "lon"]].values.astype(float)
    else:
        rng = np.random.default_rng(seed)
        coords_array = rng.integers(0, 100, size=(len(df), 2)).astype(float)
    coords_map = dict(zip(df["order_id"].tolist(), [tuple(x) for x in coords_array]))
    return coords_array, coords_map

def greedy_tsp_with_kdtree(order_ids, coords_array):
    """
    Greedy nearest-neighbor TSP using cKDTree to avoid storing full distance matrix.
    Returns route (list of order_ids with return to start) and total distance.
    """
    n = len(order_ids)
    if n == 0:
        return [], 0.0

    tree = cKDTree(coords_array)
    visited = np.zeros(n, dtype=bool)
    route_idx = []
    current_idx = 0  # start at first order in the list
    visited[current_idx] = True
    route_idx.append(current_idx)
    total_dist = 0.0

    while len(route_idx) < n:
        k = 5
        found = False
        while not found:
            k = min(k, n)
            dists, idxs = tree.query(coords_array[current_idx], k=k+1)  # include itself
            # normalize shapes
            if np.isscalar(idxs):
                idxs = np.array([idxs])
                dists = np.array([dists])
            # Try returned neighbors (skip self)
            for dist, idx in zip(dists[1:], idxs[1:]):
                if not visited[int(idx)]:
                    next_idx = int(idx)
                    total_dist += float(dist)
                    visited[next_idx] = True
                    route_idx.append(next_idx)
                    current_idx = next_idx
                    found = True
                    break
            if not found:
                if k >= n:
                    unvisited = np.where(~visited)[0]
                    if len(unvisited) == 0:
                        found = True
                        break
                    dists_linear = np.linalg.norm(coords_array[current_idx] - coords_array[unvisited], axis=1)
                    argmin = np.argmin(dists_linear)
                    next_idx = int(unvisited[argmin])
                    total_dist += float(dists_linear[argmin])
                    visited[next_idx] = True
                    route_idx.append(next_idx)
                    current_idx = next_idx
                    found = True
                else:
                    k = min(k * 2, n)

    # return to start
    start_coord = coords_array[route_idx[0]]
    last_coord = coords_array[route_idx[-1]]
    total_dist += float(np.linalg.norm(last_coord - start_coord))

    route = [order_ids[i] for i in route_idx]
    route.append(route[0])
    return route, total_dist

# Main flow

df = load_dataframe(uploaded_file, use_local)

if df is None:
    st.info("Upload a CSV or enable 'Use local df_orders.csv' with a local file present.")
    st.stop()

st.write("Columns:", df.columns.tolist())
st.write("Total rows in file:", len(df))

if run_button:
    with st.spinner("Preparing data..."):
        df_prepared = prepare_dataframe(df)
        if df_prepared is None:
            st.stop()

    # sample for visualization/speed
    df_sample = sample_dataframe(df_prepared, sample_size)
    st.write(f"Using {len(df_sample)} orders for route simulation (sampled).")

    # build coords and order lists
    coords_array, coords_map = generate_coords(df_sample)
    order_ids = df_sample["order_id"].tolist()
    order_cost = dict(zip(df_sample["order_id"], df_sample["processing_time_min"]))

    # selection based on capacity
    selected_orders, total_cost = select_orders(order_ids, order_cost, max_capacity)
    st.write(f"Selected {len(selected_orders)} orders (Total processing time: {total_cost:.1f} units)")

    if len(selected_orders) == 0:
        st.warning("No orders selected under current capacity. Increase capacity or sample size.")
    else:
        # restrict to selected orders
        sel_order_ids = selected_orders
        sel_coords_array = np.vstack([coords_map[oid] for oid in sel_order_ids])

        with st.spinner("Computing route (greedy nearest neighbor)..."):
            route, total_distance = greedy_tsp_with_kdtree(sel_order_ids, sel_coords_array)

        st.success(f"Route computed — total distance ≈ {total_distance:.2f} units")
        st.write("Route (first 20 shown):")
        st.write(route[:20])

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        # plot all sampled orders as gray
        ax.scatter(coords_array[:, 0], coords_array[:, 1], s=30, color="lightgray", label="Sampled Orders")
        # highlight selected
        sel_coords_full = np.array([coords_map[oid] for oid in sel_order_ids])
        ax.scatter(sel_coords_full[:, 0], sel_coords_full[:, 1], s=60, color="blue", label="Selected Orders")

        # route line coordinates
        route_coords = np.array([coords_map[oid] for oid in route])
        ax.plot(route_coords[:, 0], route_coords[:, 1], '-o', color="red", linewidth=2, label="Route")

        # annotate a few nodes
        for i, oid in enumerate(route):
            if i < 20:
                ax.text(route_coords[i, 0], route_coords[i, 1] + 1.5, f"{i+1}", fontsize=8, color="red", ha="center")

             # Determine axis labels dynamically
        if "latitude" in df.columns and "longitude" in df.columns:
            x_label, y_label = "Longitude", "Latitude"
        elif "lat" in df.columns and "lon" in df.columns:
            x_label, y_label = "Lon", "Lat"
        else:
            x_label, y_label = "Generated X", "Generated Y"

        ax.set_title("DelivTrack: Delivery Route Visualization🚚")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(alpha=0.4)
        st.pyplot(fig)

st.markdown("---")
st.markdown(
    "💡Tips: Increase 'Sample size' for more accurate visualization (but higher Memory/CPU), "
    "or tune 'Max capacity' to select more/less orders."
)
