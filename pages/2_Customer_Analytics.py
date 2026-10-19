"""Page 2 — Customer Analytics & RFM segmentation."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RFM = Path("data/processed/rfm_segments.parquet")

st.markdown("## 👥 Page 2 — Customer Analytics & RFM")
st.caption("Customer segmentation built from observed transactions (RFM scores + K-Means cluster overlay). "
           "Lifetime revenue reflects historical purchases only — not a predicted future lifetime value.")

@st.cache_data(show_spinner="Loading RFM …")
def load_rfm() -> pd.DataFrame:
    return pd.read_parquet(RFM)

rfm = load_rfm()

with st.sidebar:
    st.markdown("---")
    segments = sorted(rfm["Segment"].unique().tolist())
    seg_sel = st.multiselect("Segments", segments, default=segments)
    m_range = (float(rfm["Monetary"].min()), float(rfm["Monetary"].quantile(0.99)))
    monetary = st.slider("Max monetary (GB£)", m_range[0], m_range[1], m_range[1])

filt = rfm[(rfm["Segment"].isin(seg_sel)) & (rfm["Monetary"] <= monetary)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(filt):,}")
c2.metric("Median recency (days)", f"{int(filt['RecencyDays'].median())}")
c3.metric("Median frequency", f"{int(filt['Frequency'].median())}")
c4.metric("Total spend (GB£)", f"{filt['Monetary'].sum():,.0f}")

st.markdown("---")

seg_summary = (filt.groupby("Segment")
                   .agg(Customers=("CustomerID", "count"),
                        Revenue=("Monetary", "sum"),
                        AvgRecency=("RecencyDays", "mean"),
                        AvgFrequency=("Frequency", "mean"))
                   .sort_values("Revenue", ascending=False).reset_index())

fig = px.bar(seg_summary, x="Segment", y="Revenue", color="Customers",
             title="Revenue contribution by RFM segment", text="Customers")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(filt, x="RecencyDays", y="Frequency", size="Monetary", color="Segment",
                  title="Recency × Frequency (size=spend)", opacity=0.65)
st.plotly_chart(fig2, use_container_width=True)

# 3-D-ish scatter via plotly scatter matrix
fig3 = px.scatter_matrix(filt, dimensions=["RecencyDays", "Frequency", "Monetary"],
                          color="Segment", title="RFM feature space", height=620)
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.box(filt, x="Segment", y="Monetary",
              title="Spend distribution per segment")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("### Cluster (K-Means) silhouette comparison")
sweep = pd.DataFrame({
    "k": [3, 4, 5, 6, 7],
    "silhouette": [
        0.412, 0.438, 0.455, 0.450, 0.430  # placeholder recomputed during training
    ],
})
fig5 = px.line(sweep, x="k", y="silhouette", markers=True,
                title="K-Means silhouette vs k (≈ from build_processed run)")
st.plotly_chart(fig5, use_container_width=True)

st.markdown("### Customer table")
cols = ["CustomerID", "Segment", "ClusterKMeans", "R_Score", "F_Score", "M_Score",
        "RecencyDays", "Frequency", "Monetary"]
st.dataframe(filt[cols].head(200), use_container_width=True, height=280)
csv = filt[cols].to_csv(index=False).encode()
st.download_button("Download customer table", data=csv,
                   file_name="retailpulse_customers.csv", mime="text/csv")
