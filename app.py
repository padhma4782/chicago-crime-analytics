# streamlit_app.py
# Multi-page Streamlit web application for Chicago Crime Analytics

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# App Config

st.set_page_config(
    page_title="Chicago Crime Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚓 Chicago Crime Analytics Dashboard")

# --------------------------------------------------
# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("cluster_data.csv")

df = load_data()
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# --------------------------------------------------
# Load Trained K-Means Model + Scaler (USED IN APP)
# --------------------------------------------------
import joblib

@st.cache_resource
def load_kmeans_bundle():
    try:
        bundle = joblib.load("crime_zone_kmeans_bundle.pkl")
        return bundle
    except Exception:
        return None

bundle = load_kmeans_bundle()

# If model exists and crime_zone not present, COMPUTE it using the model
if bundle is not None and 'crime_zone' not in df.columns:
    kmeans = bundle['model']
    scaler = bundle['scaler']
    features = bundle.get('features', [
    'Latitude', 'Longitude'
])

    X = df[features].dropna()
    X_scaled = scaler.transform(X)

    df.loc[X.index, 'crime_zone'] = kmeans.predict(X_scaled)


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Geographic Crime Heatmap",
        "Crime Zones (K-Means)",
        "Temporal Pattern Analysis",
        "Dimensionality Reduction (PCA & t-SNE)"
    ]
)

# --------------------------------------------------
# 1️⃣ Overview Page
# --------------------------------------------------
if page == "Overview":
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Unique Crime Types", df.filter(like="primary_").sum(axis=0).astype(bool).sum())
    col3.metric("Time Span", f"{df['Year'].min()} - {df['Year'].max()}")

    st.dataframe(df.head(100))

# --------------------------------------------------
# 2️⃣ Geographic Crime Heatmap
# --------------------------------------------------
elif page == "Geographic Crime Heatmap":
    st.subheader("Geographic Crime Heatmap")

    m = folium.Map(location=[41.88, -87.63], zoom_start=11)

    for _, row in df.sample(min(5000, len(df)), random_state=42).iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=2,
            color='red',
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

    st_folium(m, width=1200, height=700)

# --------------------------------------------------
# 3️⃣ Crime Zones (K-Means)
# --------------------------------------------------
# --------------------------------------------------
# 3️⃣ Crime Zones (K-Means)
# --------------------------------------------------
elif page == "Crime Zones (K-Means)":

    st.subheader("🗺️ Crime Zones with K-Means Clustering")

    if 'crime_zone' not in df.columns:
        st.error("Crime zone labels not found in dataset.")

    else:

        # Remove invalid coordinates / clusters
        map_df = df.dropna(
            subset=['Latitude', 'Longitude', 'crime_zone']
        ).copy()

        map_df['crime_zone'] = map_df['crime_zone'].astype(int)

        # --------------------------------------------------
        # SIDEBAR FILTERS
        # --------------------------------------------------

        st.sidebar.markdown("### 🎯 Map Controls")

        clusters = sorted(map_df['crime_zone'].unique())

        selected_clusters = st.sidebar.multiselect(
            "Select Crime Zones",
            clusters,
            default=clusters
        )

        filtered_df = map_df[
            map_df['crime_zone'].isin(selected_clusters)
        ]

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        total_crimes = len(filtered_df)
        total_zones = filtered_df['crime_zone'].nunique()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Crimes",
            f"{total_crimes:,}"
        )

        c2.metric(
            "Crime Zones",
            total_zones
        )

        if total_zones > 0:
            largest_zone = (
                filtered_df['crime_zone']
                .value_counts()
                .index[0]
            )

            largest_zone_count = (
                filtered_df['crime_zone']
                .value_counts()
                .iloc[0]
            )
        else:
            largest_zone = "-"
            largest_zone_count = 0

        c3.metric(
            "Largest Zone",
            f"Zone {largest_zone}"
        )

        c4.metric(
            "Largest Zone Crimes",
            f"{largest_zone_count:,}"
        )

        # --------------------------------------------------
        # CLUSTER COLORS
        # --------------------------------------------------

        cluster_colors = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabed4",
            "#469990"
        ]

        # --------------------------------------------------
        # CREATE MAP
        # --------------------------------------------------

        m = folium.Map(
            location=[41.8781, -87.6298],
            zoom_start=10,
            tiles="CartoDB positron",
            control_scale=True
        )

        # --------------------------------------------------
        # CREATE FEATURE GROUP FOR EACH CLUSTER
        # --------------------------------------------------

        for cluster in selected_clusters:

            cluster_df = filtered_df[
                filtered_df['crime_zone'] == cluster
            ]

            color = cluster_colors[
                cluster % len(cluster_colors)
            ]

            feature_group = folium.FeatureGroup(
                name=f"Crime Zone {cluster}"
            )

            # --------------------------------------------------
            # LIMIT POINTS ONLY FOR PERFORMANCE
            # --------------------------------------------------

            display_df = cluster_df.sample(
                min(1000, len(cluster_df)),
                random_state=42
            )

            for _, row in display_df.iterrows():

                popup_html = f"""
                <div style="width:220px">

                    <h4>🚨 Crime Zone {cluster}</h4>

                    <b>Latitude:</b>
                    {row['Latitude']:.5f}

                    <br>

                    <b>Longitude:</b>
                    {row['Longitude']:.5f}

                    <br><br>

                    <b>Year:</b>
                    {row.get('Year', 'N/A')}

                    <br>

                    <b>Hour:</b>
                    {row.get('Hour', 'N/A')}

                </div>
                """

                folium.CircleMarker(
                    location=[
                        row['Latitude'],
                        row['Longitude']
                    ],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.55,
                    weight=1,
                    popup=folium.Popup(
                        popup_html,
                        max_width=300
                    )
                ).add_to(feature_group)

            feature_group.add_to(m)

        # --------------------------------------------------
        # LEGEND
        # --------------------------------------------------

        legend_items = ""

        for cluster in selected_clusters:

            color = cluster_colors[
                cluster % len(cluster_colors)
            ]

            count = len(
                filtered_df[
                    filtered_df['crime_zone'] == cluster
                ]
            )

            legend_items += f"""
            <div style="
                margin-bottom:6px;
                font-size:14px;
            ">
                <span style="
                    display:inline-block;
                    width:14px;
                    height:14px;
                    background:{color};
                    border-radius:50%;
                    margin-right:7px;
                "></span>

                Zone {cluster}
                <span style="color:#666">
                    ({count:,})
                </span>
            </div>
            """

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;

            background: white;
            padding: 12px 16px;

            border-radius: 8px;

            box-shadow:
                0 2px 8px rgba(0,0,0,0.25);

            min-width: 150px;
        ">

            <b>Crime Zones</b>

            <hr>

            {legend_items}

        </div>
        """

        m.get_root().html.add_child(
            folium.Element(legend_html)
        )

        # --------------------------------------------------
        # LAYER CONTROL
        # --------------------------------------------------

        folium.LayerControl(
            collapsed=False
        ).add_to(m)

        # --------------------------------------------------
        # DISPLAY MAP
        # --------------------------------------------------

        st_folium(
            m,
            width=None,
            height=700,
            returned_objects=[]
        )

        # --------------------------------------------------
        # CLUSTER SUMMARY
        # --------------------------------------------------

        st.markdown("### 📊 Crime Zone Summary")

        zone_summary = (
            filtered_df
            .groupby('crime_zone')
            .size()
            .reset_index(name='Crime Count')
            .sort_values(
                'Crime Count',
                ascending=False
            )
        )

        zone_summary['Percentage'] = (
            zone_summary['Crime Count']
            / zone_summary['Crime Count'].sum()
            * 100
        ).round(2)

        st.dataframe(
            zone_summary,
            use_container_width=True,
            hide_index=True
        )
# --------------------------------------------------
# 4️⃣ Temporal Pattern Analysis
# --------------------------------------------------
elif page == "Temporal Pattern Analysis":
    st.subheader("Temporal Crime Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Crimes by Hour")
        hour_counts = df.groupby('Hour').size()
        fig, ax = plt.subplots()
        ax.plot(hour_counts.index, hour_counts.values)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Crime Count")
        st.pyplot(fig)

    with col2:
        st.write("### Crimes by Day of Week")
        day_counts = df.groupby('Day_enc').size()
        day_labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
        plt.xticks(day_counts, day_labels)
        fig, ax = plt.subplots()
        ax.bar(day_counts.index, day_counts.values)
        ax.set_xlabel("Day (Encoded)")
        ax.set_ylabel("Crime Count")
        st.pyplot(fig)

# --------------------------------------------------
# 5️⃣ Dimensionality Reduction
# --------------------------------------------------
elif page == "Dimensionality Reduction (PCA & t-SNE)":

    st.subheader("📊 Dimensionality Reduction Analysis")

    st.caption(
        "Visualization of spatial, temporal and crime-frequency "
        "patterns in the crime dataset."
    )

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    features = [
        'Latitude',
        'Longitude',
        'Hour',
        'Month',
        'Day_enc',
        'location_freq'
    ]

    # --------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = features + ['crime_zone']

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

    else:

        # --------------------------------------------------
        # PREPARE DATA
        # --------------------------------------------------

        reduction_df = df[
            required_columns
        ].dropna().copy()

        reduction_df['crime_zone'] = (
            reduction_df['crime_zone']
            .astype(int)
        )

        # --------------------------------------------------
        # SAMPLE DATA
        # --------------------------------------------------

        sample_size = min(
            10000,
            len(reduction_df)
        )

        reduction_sample = reduction_df.sample(
            n=sample_size,
            random_state=42
        )

        X = reduction_sample[features]

        zones = reduction_sample[
            'crime_zone'
        ]

        # --------------------------------------------------
        # STANDARDIZATION
        # --------------------------------------------------

        scaler_reduction = StandardScaler()

        X_scaled = scaler_reduction.fit_transform(X)

        # ==================================================
        # PCA
        # ==================================================

        st.markdown("## 🔵 PCA Analysis")

        st.write(
            "PCA reduces the six original features into "
            "principal components for visualization."
        )

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        pca = PCA(
            n_components=2
        )

        X_pca = pca.fit_transform(
            X_scaled
        )

        # Explained variance
        pc1_variance = (
            pca.explained_variance_ratio_[0]
            * 100
        )

        pc2_variance = (
            pca.explained_variance_ratio_[1]
            * 100
        )

        total_variance = (
            pc1_variance +
            pc2_variance
        )

        # --------------------------------------------------
        # PCA METRICS
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "PC1 Variance",
            f"{pc1_variance:.1f}%"
        )

        col2.metric(
            "PC2 Variance",
            f"{pc2_variance:.1f}%"
        )

        col3.metric(
            "PC1 + PC2",
            f"{total_variance:.1f}%"
        )

        # ==================================================
        # PCA SCATTER PLOT
        # ==================================================

        st.write("### PCA 2D Visualization")

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=zones,
            cmap="tab10",
            s=8,
            alpha=0.45,
            edgecolors="none"
        )

        ax.set_xlabel(
            f"PC1 ({pc1_variance:.1f}%)"
        )

        ax.set_ylabel(
            f"PC2 ({pc2_variance:.1f}%)"
        )

        ax.set_title(
            "PCA: Crime Data by K-Means Zone"
        )

        # Legend
        handles, labels = (
            scatter.legend_elements()
        )

        ax.legend(
            handles,
            labels,
            title="Crime Zone",
            loc="best"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.info(
            f"PC1 explains {pc1_variance:.1f}% "
            f"and PC2 explains {pc2_variance:.1f}% "
            f"of the variance. Together they explain "
            f"{total_variance:.1f}%."
        )

        # ==================================================
        # PCA FEATURE LOADINGS
        # ==================================================

        st.write("### 🔍 PCA Feature Loadings")

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[
                "PC1",
                "PC2"
            ],
            index=features
        )

        loadings_display = loadings.round(4)

        st.dataframe(
            loadings_display,
            use_container_width=True
        )

        st.caption(
            "Larger absolute loading values indicate "
            "a stronger contribution of a feature to "
            "the corresponding principal component."
        )

        # ==================================================
        # t-SNE
        # ==================================================

        st.markdown("## 🟣 t-SNE Analysis")

        st.write(
            "t-SNE emphasizes local similarities "
            "between crime records."
        )

        # --------------------------------------------------
        # t-SNE
        # --------------------------------------------------

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            init="pca",
            learning_rate="auto"
        )

        X_tsne = tsne.fit_transform(
            X_scaled
        )

        # ==================================================
        # t-SNE PLOT
        # ==================================================

        fig, ax = plt.subplots(
            figsize=(11, 7)
        )

        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=zones,
            cmap="tab10",
            s=8,
            alpha=0.45,
            edgecolors="none"
        )

        ax.set_xlabel(
            "t-SNE 1"
        )

        ax.set_ylabel(
            "t-SNE 2"
        )

        ax.set_title(
            "t-SNE: Crime Data by K-Means Zone"
        )

        # Legend
        handles, labels = (
            scatter.legend_elements()
        )

        ax.legend(
            handles,
            labels,
            title="Crime Zone",
            loc="best"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.info(
            f"PCA and t-SNE visualization uses "
            f"{sample_size:,} sampled crime records."
        )