
# ============================================================
# PatrolIQ - Chicago Crime Analytics Dashboard
# K-Means | DBSCAN | Hierarchical | PCA | t-SNE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

from streamlit_folium import st_folium

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster



st.set_page_config(
    page_title="PatrolIQ - Chicago Crime Analytics",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("PatrolIQ - Chicago Crime Analytics")
st.caption(
    "Unsupervised Learning Dashboard | "
    "K-Means • DBSCAN • Hierarchical Clustering • PCA • t-SNE"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("cluster_data.csv")

    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    return df


df = load_data()



required_columns = [
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Day_enc",
    "Is_Weekend",
    "location_freq",
    "Beat",
    "District",
    "Ward",
    "Community Area"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing from "
        "cluster_data.csv:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# 4. COMMON FEATURE SETS
# ============================================================


CLUSTER_FEATURES = [
    "Latitude",
    "Longitude",
    "Hour",
    "Is_Weekend",
    "location_freq"
]

K=8

PCA_FEATURES = [
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Day_enc",
    "Is_Weekend",
    "location_freq",
    "Beat",
    "District",
    "Ward",
    "Community Area"
]


TSNE_FEATURES = [
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Day_enc",
    "Is_Weekend",
    "location_freq",
    "Arrest",
    "Domestic",
    "primary_Theft",
    "primary_Battery",
    "primary_Assault",
    "primary_Robbery",
    "primary_Narcotics"
]




def safe_sample(data, n, random_state=42):

    return data.sample(
        min(n, len(data)),
        random_state=random_state
    )


def create_map():

    return folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles="OpenStreetMap"
    )


def add_points_to_map(
    map_object,
    data,
    color="red",
    radius=3,
    opacity=0.5
):

    for _, row in data.iterrows():

        if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
            continue

        folium.CircleMarker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity
        ).add_to(map_object)


# ============================================================
# 6. K-MEANS
# ============================================================

@st.cache_data
def run_kmeans(data):

    work = data[
        CLUSTER_FEATURES
    ].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(work)

    k = 8

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    work["crime_zone"] = labels

    return work, model, scaler




# ============================================================
# 8. DBSCAN
# ============================================================

@st.cache_data
def run_dbscan(data):

    sample = safe_sample(
        data,
        100000
    )

    work = sample[
        ["Latitude", "Longitude"]
    ].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(work)

    model = DBSCAN(
        eps=0.04,
        min_samples=80,
        n_jobs=-1
    )

    labels = model.fit_predict(X_scaled)

    work["dbscan_cluster"] = labels

    return work, model


# ============================================================
# 9. HIERARCHICAL CLUSTERING
# ============================================================

@st.cache_data
def run_hierarchical(data):

    sample = safe_sample(
        data,
        3000
    )

    work = sample[
        ["Latitude", "Longitude"]
    ].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(work)

    linkage_matrix = linkage(
        X_scaled,
        method="ward"
    )

    return work, linkage_matrix


# ============================================================
# 10. PCA
# ============================================================

@st.cache_data
def run_pca(data):

    work = data[
        PCA_FEATURES
    ].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(work)

    pca = PCA(
        n_components=3,
        random_state=42
    )

    transformed = pca.fit_transform(X_scaled)

    result = pd.DataFrame(
        transformed,
        columns=[
            "PC1",
            "PC2",
            "PC3"
        ],
        index=work.index
    )

    return result, pca


# ============================================================
# 11. t-SNE
# ============================================================

@st.cache_data
def run_tsne(data):

    sample = safe_sample(
        data,
        10000
    )

    work = sample[
        TSNE_FEATURES
    ].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(work)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )

    embedding = tsne.fit_transform(X_scaled)

    result = pd.DataFrame(
        embedding,
        columns=[
            "TSNE1",
            "TSNE2"
        ],
        index=work.index
    )

    return result


# ============================================================
# 12. SIDEBAR
# ============================================================

st.sidebar.title("PatrolIQ")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Crime Heatmap",
        "K-Means Clustering",
        "DBSCAN Clustering",
        "Hierarchical Clustering",
        "PCA Analysis",
        "t-SNE Analysis",
        "Temporal Analysis"
    ]
)


# ============================================================
# 13. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Features",
        len(df.columns)
    )

    if "Year" in df.columns:

        col3.metric(
            "Year",
            f"{df['Year'].min()} - {df['Year'].max()}"
        )

    else:

        col3.metric(
            "Year",
            "N/A"
        )

    if "primary_Theft" in df.columns:

        theft_count = int(
            df["primary_Theft"].sum()
        )

    else:

        theft_count = 0

    col4.metric(
        "Theft Records",
        f"{theft_count:,}"
    )

    st.divider()

    st.subheader("Dataset Sample")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(df[col].isna().sum())
            for col in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


# ============================================================
# 14. CRIME HEATMAP
# ============================================================

elif page == "Crime Heatmap":

    st.header("Geographic Crime Heatmap")

    st.write(
        "Geographic distribution of reported crime incidents."
    )

    sample_size = st.slider(
        "Number of points",
        min_value=1000,
        max_value=min(10000, len(df)),
        value=min(5000, len(df)),
        step=1000
    )

    map_data = safe_sample(
        df,
        sample_size
    )

    m = create_map()

    add_points_to_map(
        m,
        map_data,
        color="red",
        radius=2,
        opacity=0.25
    )

    st_folium(
        m,
        width=None,
        height=700
    )


# ============================================================
# 15. K-MEANS
# ============================================================

elif page == "🎯 K-Means Clustering":
    pass
# ============================================================
# 16. DBSCAN
# ============================================================

elif page == "🔎 DBSCAN Clustering":

    st.header("🔎 DBSCAN Crime Hotspots")

    st.write(
        "DBSCAN identifies dense geographic crime hotspots "
        "and separates noise/outlier locations."
    )

    st.info(
        "DBSCAN configuration: eps = 0.04, "
        "min_samples = 80"
    )

    with st.spinner(
        "Running DBSCAN on up to 100,000 records..."
    ):

        db_data, dbscan = run_dbscan(df)

    labels = db_data["dbscan_cluster"]

    total_clusters = len(
        set(labels)
        - {-1}
    )

    noise_count = int(
        (labels == -1).sum()
    )

    noise_pct = (
        noise_count /
        len(labels) *
        100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Clusters",
        total_clusters
    )

    col2.metric(
        "Noise Points",
        f"{noise_count:,}"
    )

    col3.metric(
        "Noise %",
        f"{noise_pct:.2f}%"
    )

    st.divider()

    # --------------------------------------------------------
    # Cluster distribution
    # --------------------------------------------------------

    st.subheader(
        "DBSCAN Cluster Distribution"
    )

    counts = (
        labels
        .value_counts()
        .sort_index()
    )

    display_counts = counts.copy()

    display_counts.index = [
        "Noise (-1)"
        if x == -1
        else f"Cluster {x}"
        for x in display_counts.index
    ]

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        display_counts.index.astype(str),
        display_counts.values
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Records")
    ax.set_title(
        "DBSCAN Crime Cluster Distribution"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Geographic plot
    # --------------------------------------------------------

    st.subheader(
        "DBSCAN Crime Hotspots"
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    noise = db_data[
        db_data["dbscan_cluster"] == -1
    ]

    hotspots = db_data[
        db_data["dbscan_cluster"] != -1
    ]

    ax.scatter(
        noise["Longitude"],
        noise["Latitude"],
        s=5,
        alpha=0.25,
        label="Noise"
    )

    if len(hotspots) > 0:

        scatter = ax.scatter(
            hotspots["Longitude"],
            hotspots["Latitude"],
            c=hotspots["dbscan_cluster"],
            cmap="tab20",
            s=5,
            alpha=0.6
        )

        plt.colorbar(
            scatter,
            ax=ax,
            label="DBSCAN Cluster"
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_title(
        "DBSCAN Crime Hotspots"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Top clusters
    # --------------------------------------------------------

    st.subheader(
        "Top 10 DBSCAN Hotspots"
    )

    top_clusters = (
        db_data[
            db_data["dbscan_cluster"] != -1
        ]
        .groupby("dbscan_cluster")
        .size()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index(
            name="Crime Records"
        )
    )

    st.dataframe(
        top_clusters,
        use_container_width=True
    )


# ============================================================
# 17. HIERARCHICAL CLUSTERING
# ============================================================

elif page == "🌳 Hierarchical Clustering":

    st.header(
        "🌳 Hierarchical Clustering"
    )

    st.write(
        "Ward hierarchical clustering is applied to a "
        "3,000-record sample using Latitude and Longitude."
    )

    with st.spinner(
        "Calculating hierarchical clustering..."
    ):

        hier_data, linkage_matrix = (
            run_hierarchical(df)
        )

    # --------------------------------------------------------
    # Dendrogram
    # --------------------------------------------------------

    st.subheader(
        "Hierarchical Clustering Dendrogram"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        p=5,
        ax=ax
    )

    ax.set_title(
        "Hierarchical Clustering Dendrogram "
        "(Crime Zones)"
    )

    ax.set_xlabel(
        "Sample Index"
    )

    ax.set_ylabel(
        "Distance"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Geographic visualization
    # --------------------------------------------------------

    st.subheader(
        "Hierarchical Crime Zones"
    )

    n_clusters = st.slider(
        "Number of hierarchical clusters",
        min_value=2,
        max_value=10,
        value=8
    )

    cluster_labels = fcluster(
        linkage_matrix,
        t=n_clusters,
        criterion="maxclust"
    )

    plot_data = hier_data.copy()

    plot_data[
        "hier_cluster"
    ] = cluster_labels

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    scatter = ax.scatter(
        plot_data["Longitude"],
        plot_data["Latitude"],
        c=plot_data["hier_cluster"],
        cmap="tab10",
        s=6,
        alpha=0.6
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_title(
        "Hierarchical Crime Zones"
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 18. PCA
# ============================================================

elif page == "📉 PCA Analysis":

    st.header(
        "📉 PCA - Principal Component Analysis"
    )

    st.write(
        "PCA reduces the 11-dimensional feature space "
        "to three principal components."
    )

    with st.spinner(
        "Calculating PCA..."
    ):

        pca_data, pca = run_pca(df)

    explained = (
        pca.explained_variance_ratio_
    )

    cumulative = explained.cumsum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PC1",
        f"{explained[0] * 100:.2f}%"
    )

    col2.metric(
        "PC2",
        f"{explained[1] * 100:.2f}%"
    )

    col3.metric(
        "PC3",
        f"{explained[2] * 100:.2f}%"
    )

    st.subheader(
        "Explained Variance"
    )

    variance_df = pd.DataFrame({
        "Component": [
            "PC1",
            "PC2",
            "PC3"
        ],
        "Explained Variance (%)": [
            x * 100
            for x in explained
        ],
        "Cumulative Variance (%)": [
            x * 100
            for x in cumulative
        ]
    })

    st.dataframe(
        variance_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PCA scatter
    # --------------------------------------------------------

    st.subheader(
        "PCA 2D Visualization"
    )

    plot_data = pca_data

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.scatter(
        plot_data["PC1"],
        plot_data["PC2"],
        s=4,
        alpha=0.5
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(
        "PCA - PC1 vs PC2"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PCA loadings
    # --------------------------------------------------------

    st.subheader(
        "Top Features Contributing to PCA"
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=[
            "PC1",
            "PC2",
            "PC3"
        ]
    )

    loadings["importance"] = (
        loadings
        .abs()
        .sum(axis=1)
    )

    top5 = (
        loadings
        .sort_values(
            "importance",
            ascending=False
        )
        .head(5)
    )

    st.dataframe(
        top5,
        use_container_width=True
    )


# ============================================================
# 19. t-SNE
# ============================================================

elif page == "🧬 t-SNE Analysis":

    st.header(
        "🧬 t-SNE Crime Pattern Visualization"
    )

    st.write(
        "t-SNE projects the selected crime features "
        "into two dimensions to visualize local patterns."
    )

    st.info(
        "Configuration: 10,000-record sample, "
        "perplexity = 30, learning rate = 200, "
        "max_iter = 1000"
    )

    run_button = st.button(
        "▶ Run t-SNE"
    )

    if run_button:

        st.session_state[
            "run_tsne"
        ] = True

    if st.session_state.get(
        "run_tsne",
        False
    ):

        with st.spinner(
            "Running t-SNE. This may take some time..."
        ):

            tsne_data = run_tsne(df)

        # ----------------------------------------------------
        # If crime_zone exists, add K-Means labels
        # ----------------------------------------------------

        try:

            km_data, _, _ = run_kmeans(df)

            common_index = (
                tsne_data.index
                .intersection(
                    km_data.index
                )
            )

            tsne_plot = tsne_data.loc[
                common_index
            ].copy()

            tsne_plot[
                "crime_zone"
            ] = km_data.loc[
                common_index,
                "crime_zone"
            ]

        except Exception:

            tsne_plot = tsne_data.copy()

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        st.subheader(
            "t-SNE 2D Visualization"
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        if "crime_zone" in tsne_plot.columns:

            scatter = ax.scatter(
                tsne_plot["TSNE1"],
                tsne_plot["TSNE2"],
                c=tsne_plot["crime_zone"],
                cmap="tab10",
                s=6,
                alpha=0.6
            )

            plt.colorbar(
                scatter,
                ax=ax,
                label="K-Means Crime Zone"
            )

        else:

            ax.scatter(
                tsne_plot["TSNE1"],
                tsne_plot["TSNE2"],
                s=6,
                alpha=0.6
            )

        ax.set_xlabel(
            "t-SNE 1"
        )

        ax.set_ylabel(
            "t-SNE 2"
        )

        ax.set_title(
            "t-SNE Visualization of Crime Patterns"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.success(
            f"t-SNE completed for "
            f"{len(tsne_data):,} records."
        )


# ============================================================
# 20. TEMPORAL ANALYSIS
# ============================================================

elif page == "📈 Temporal Analysis":

    st.header(
        "📈 Temporal Crime Pattern Analysis"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Crimes by hour
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Crimes by Hour"
        )

        hour_counts = (
            df.groupby("Hour")
            .size()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            hour_counts.index,
            hour_counts.values,
            marker="o"
        )

        ax.set_xlabel(
            "Hour of Day"
        )

        ax.set_ylabel(
            "Crime Count"
        )

        ax.set_title(
            "Crime Distribution by Hour"
        )

        ax.set_xticks(
            range(0, 24)
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Crimes by day
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Crimes by Day of Week"
        )

        day_counts = (
            df.groupby("Day_enc")
            .size()
            .sort_index()
        )

        day_labels = [
            "Sun",
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat"
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            day_counts.index,
            day_counts.values
        )

        ax.set_xlabel(
            "Day of Week"
        )

        ax.set_ylabel(
            "Crime Count"
        )

        ax.set_title(
            "Crime Distribution by Day"
        )

        ax.set_xticks(
            range(7)
        )

        ax.set_xticklabels(
            day_labels
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "PatrolIQ | Chicago Crime Analytics"
)

st.sidebar.caption(
    "Unsupervised ML Dashboard"
)