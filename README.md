# Data-Driven Geospatial Forensics of Infrastructure Deformation Using InSAR Time Series and Unsupervised Learning

## Research Question
**What moved, when, and why?**

## Aim
Provide policy-relevant insights to infrastructure stakeholders—particularly those responsible for Sweden’s railway and highway systems—by identifying and explaining ground motion patterns linked to natural hazards. The goal is to support preventative maintenance, reduce financial risk, and avoid human loss due to deformation-induced failure.

## Objectives
- Decompose the deformation into meaningful components (e.g., linear trends, seasonal cycles, abrupt steps).
- Attribute these components to potential drivers such as hydroclimatic forcing, land cover change, and underlying geology.
- Develop an interpretable, data-driven framework that explains deformation behavior using unsupervised learning and post-hoc correlation analysis.

## Output
A modular deformation modeling framework with:
- A breakdown of InSAR-observed motion into physical categories,
- Anomaly detection linked to expected motion behavior,
- Environmental correlation explaining “why” deformation occurred,
- Risk maps and deformation taxonomies tailored for infrastructure screening and early warning.

---

## 1. Context: What is InSAR Deformation Modeling?

InSAR time series track ground displacement at specific pixels across repeated radar acquisitions (e.g., every 6 or 12 days with Sentinel-1). These observations are typically visualized as a displacement time series per pixel.

We are interested in modeling these time series not just as linear trends but as possibly **nonlinear or event-driven signals**. For example:

- **Step-like motion**: A sudden shift due to a discrete event (e.g., landslide, earthquake). The time series shows constant trends before and after, with a jump at some unknown time `tk`.
- **Breakpoint behavior**: A change in deformation rate, such as acceleration or deceleration in subsidence or uplift.
- **Seasonal behavior**: Oscillations driven by snowmelt, groundwater dynamics, or vegetation cycles.

The challenge: The timing and nature of such changes are **not known in advance**.

So we aim to:
- Test plausible deformation models for each pixel or zone (linear, step, breakpoint, seasonal).
- Determine which model fits best and when the deformation changed.
- Attribute these changes to potential triggers like **rainfall, vegetation loss, or slope instability**.

---

## 2. Why Does This Matter in InSAR?

Modeling realistic deformation dynamics is critical for:
- Capturing **nonlinear deformation** patterns,
- Detecting **event-driven failures** (e.g., precipitation-triggered slope instability),
- Avoiding **overfitting** and maintaining interpretability in large datasets,
- Designing **early warning systems** for infrastructure hazards.

This project builds on and compares five key methods that tackle deformation classification, anomaly detection, and spatio-temporal clustering of InSAR time series.

---

## 3. Literature Foundations

This section outlines five highly relevant prior studies. Each presents a unique angle on deformation modeling and helps inspire variants or components of the proposed work.

---

### [1] Chang et al. (2017): Nationwide Railway Monitoring Using Model Library + Hypothesis Testing

- Develops a **library of deformation models**:
  - Linear trend
  - Step at time `τ`
  - Breakpoint in trend (velocity change)
  - Seasonal (e.g., thermal expansion)

- Applies **multiple hypothesis testing** per short arc of the network.

#### Why it’s relevant to our case:
- Their **“short-arc” approach** is ideal for identifying local anomalies (e.g., slope failures from rainfall).
- Their statistical testing framework minimizes overfitting and false alarms.
- The model library can be extended with **hydroclimate-responsive models** (e.g., lagged rainfall responses).

#### Potential Modifications:
- Add rainfall-triggered or SWE-induced steps/breakpoints to the model space.
- Use hydrological variables (e.g., NDVI, soil saturation, snowmelt) to define timing hypotheses.
- Apply **spatial clustering** to model-fit results to map coherent failure zones.

---

### [2] Schlögl et al. (2021): Clean Decomposition + Attribution + Clustering

- Applies robust preprocessing:
  - Outlier cleaning
  - LOESS smoothing
  - Temporal bias correction (e.g., temperature)

- Performs **STL decomposition** to isolate:
  - Trend
  - Seasonal components
  - Residuals

- Applies clustering to group time series by similarity.

#### Why it’s relevant:
- Strong on separating **signal from noise**.
- Decomposition is useful for **seasonal or cyclic processes** like snowmelt.
- Bias correction and attribution open the door to post-hoc environmental correlations.

#### Potential Modifications:
- Swap temperature bias correction for precipitation or SWE-driven patterns.
- Use time-lagged correlation or **Granger causality** between deformation and environmental variables.
- Combine STL decomposition with **hypothesis testing** from Chang et al. [1] to refine physical model detection.

---

### [3] Prasanthi et al. (2025): Deep Embedded Clustering of Trend Classes

1. **Post-processing InSAR series**:
   - Regrids scatter points onto a uniform pixel grid to align Ascending/Descending tracks.
   - Converts LOS to vertical & horizontal components using geometric formulas (Eqs. 2–3).
   - Applies smoothing (LOESS), outlier filtering.

2. **Dimensionality Reduction with PCAnet**:
   - Uses two-layer convolutional PCAnet to reduce redundancy while preserving spatial correlation.
   - Addresses PCA’s tendency to discard local anomalies.

3. **Autoencoder-Based Feature Compression**:
   - Trains an undercomplete autoencoder to extract latent representations from PCAnet output.
   - Learns nonlinear representations tuned for deformation structure.

4. **Spatio-Temporal Clustering**:
   - Applies sliding windows to latent time series to identify behavioral similarities over time.
   - Builds a spatial similarity matrix from co-clustering frequency.
   - Performs second clustering pass to delineate spatial zones of shared deformation behavior.

5. **Deep Embedded Clustering (DEC)**:
   - Jointly optimizes feature learning and clustering.
   - Uses Student-t kernel soft assignments, entropy-weighted target distributions (Eq. 14–16).
   - Minimizes KL divergence + reconstruction loss to form deformation-type-aware clusters.

6. **Osprey Optimization Algorithm (OOA)**:
   - Two-stage global+local metaheuristic for centroid and hyperparameter tuning.
   - “Explore” with population-based updates (Eqs. 20–23), then “Exploit” (Eqs. 24–26).

7. **Trend Extraction**:
   - Applies regression to each final cluster to extract long-term deformation trend.

---

### [4] Kuzu et al. (2023): Building Displacement Anomalies from InSAR Time-Series (EGMS)

Detects building displacement anomalies using EGMS PS time-series **without any need for labeled ground-truth data**.

🔗 **Check out their platform**:  
https://maps.co/map/63fc2919d50c0033105031ocd4dd22a

#### Data Preparation
- Uses PS points from the European Ground Motion Service over Rome.
- Associates each PS to a building footprint.

#### Preprocessing
- Normalize each PS time-series to zero mean/unit variance.
- Smooth time-series with a Hann window (reduces random noise but preserves trend/step anomalies).
- Randomly permute time steps to make the model sensitive to true temporal structure and not just overall shape.

#### Model: LSTM Autoencoder
- An encoder LSTM compresses the permuted time-series.
- A decoder LSTM reconstructs the correctly ordered, smoothed time-series.

#### Loss Function: Soft-DTW
- Measures similarity between original and reconstructed time-series.
- Allows **time shifts** (important for deformation signals that don't align perfectly across space).

#### Anomaly Detection
- If a PS instance has a **high reconstruction error** (based on soft-DTW), it's flagged as anomalous.
- Anomalies are then classified into:
  - **Trend anomalies**
  - **Noise anomalies**
  - **Step anomalies**
- Classification is done using proximity to cluster centers in soft-DTW space.

#### Evaluation
- Benchmarked against:
  - Random Forest
  - Vanilla LSTM Autoencoders
- Outperforms both, especially in identifying **trend-type anomalies**.

#### Strengths:
- **Modern and elegant**: The random permutation trick + soft-DTW loss forces the model to learn true temporal dependencies.
- **Realistic**: Uses EGMS data; no reliance on toy examples.
- Handles both **gradual** and **sudden** motions.
- Applicable for large-scale, label-free infrastructure screening.

#### Weaknesses / Risks:
- Validation based on **synthetic anomalies** — may not fully reflect natural failure dynamics.
- Heavy smoothing (Hann window) suppresses high-frequency signals (e.g., vibration, microfailures).
- Detected anomalies require **manual or auxiliary analysis** to interpret causes.

#### In our case (railway monitoring):
- Highly relevant: detecting **creep or subtle precursors** in embankments before visible failure.
- We might need to **adjust smoothing** to retain sharper motion events (e.g., rapid collapse after rain).

---

### [5] Bai et al. (2025): KCC-LSTM — Cluster-Predict Framework for Deformation Series

Presents a method combining **temporal clustering**, **spatial grouping**, and **per-cluster LSTM modeling**.

#### Main Idea
- Detect different types of deformation (linear, periodic, nonlinear) **without labels**.
- Use shape-aware clustering + LSTM forecasting per group.
- Evaluate accuracy using RMSE, MAE, correlation.

#### Model Steps:
1. **InSAR Preprocessing**
   - Correct APS
   - Filter for reliable PS points

2. **K-shape Clustering**
   - Shape-based time series clustering algorithm.
   - Captures **temporal similarity** (pattern shape, not magnitude).

3. **Spatial Clustering**
   - Apply DBSCAN or Euclidean distance-based grouping.
   - Combine spatial and temporal constraints → **KCC: K-shape + Coordinate Clustering**.

4. **LSTM Training per Cluster**
   - Train one LSTM per deformation behavior cluster.
   - Learns to predict future displacements for that type.

5. **Evaluation Metrics**
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - Pearson Correlation Coefficient

#### Relevance to our study:
- The **per-cluster modeling** matches our goal of classifying anomaly *types*, not just flags.
- Temporal + spatial linkage enforces **interpretable behavioral groups**.
- Works well even on sparse deformation data like EGMS.

---

## 4. Implementation Options

We propose three implementation ideas, all building on the literature reviewed above.

---

### Idea #1: Combine Chang et al. [1] + Schlögl et al. [2]

#### Goal
Detect and decompose deformation patterns in a failed railway segment, compare them to other high-deformation areas, and evaluate whether failure-linked deformation has **unique temporal or environmental features**.

#### Step-by-Step Framework

##### Step 1: Data Preparation
- **InSAR Data**:
  - Use ready-to-go LOS motion or decomposed vertical/horizontal components.
  - Build “short arcs” for local gradient detection (à la Chang et al.).

- **Environmental Data**:
  - Daily/cumulative precipitation
  - Max/min temperature
  - Soil moisture / SWE
  - Land cover/NDVI (via Google Earth Engine)
  - Geological type, substrate, DEM slope/aspect
  - Known failure inventory (e.g., Hudiksvall, Stenungsund)

##### Step 2: Time Series Modeling
- Fit Chang et al.’s model library at each PS point or short arc:
  - H₀: linear trend
  - H₁: step at τ₁
  - H₂: breakpoint (velocity change at τ₂)
  - H₃: seasonal term (e.g., temperature-driven)
  - H₄+: combinations or composite behaviors

- Select the best-fit model via likelihood ratio (e.g., Eq. 11 in Chang et al.).

Outputs:
- Best-fit model class (trend, step, etc.)
- Timing of event (τ)
- Pre- and post-event rates
- Model residuals

##### Step 3: Attribution Analysis

- **Time Series-Level Correlation**:
  - Cross-correlate deformation with:
    - Rainfall
    - Temperature
    - NDVI change
    - Soil/geology/slope

- **Machine Learning Augmentation**:
  - Train classifier (e.g., random forest) on:
    - Model types (from Step 2)
    - Rain stats (mean, max, sum)
    - Temperature signals
    - Geology/slope/vegetation type

##### Step 4: Comparative Analysis

- **Cross-site Comparison**:
  - Compare failed vs. non-failed segments with similar motion magnitude.
  - Look at:
    - Deformation onset timing
    - Acceleration patterns
    - Event triggers
    - Residual errors from model fits

- **Clustering**:
  - Cluster points based on:
    - Time series behavior
    - Environmental forcing
    - Morphological context

##### Step 5: Interpretation

- Are failed zones statistically distinct in:
  - Motion type?
  - Timing?
  - Environmental correlation?

- Use tools like:
  - SHAP or permutation importance
  - UMAP for 2D visualization
  - Risk factor heatmaps

##### Output
- Deformation behavior taxonomy
- Risk indicator set for future failures
- Forensic reconstruction of E6-type events

---

### Idea #2: Modify Kuzu et al. (2023) — Ground Motion Anomaly Detection (Soft-DTW + DEM, Post-hydro Analysis)

#### Step 0 — Data Preparation

- **InSAR Time Series Collection**
  - LOS motion from ascending and descending tracks from EGMS.
  - Aligned in time and space, ideally on a common pixel grid.

- **DEM Extraction**
  - Clip and downsample DEM to the InSAR grid.
  - Use DEM as an **input channel** to help model learn topography-correlated atmospheric noise.

- **Environmental Variables (Post-analysis only)**
  - Precipitation, SWE, temperature, NDVI/NDWI.
  - Interpolate/extract at each InSAR pixel, but **do not feed into the model**.
  - These will be used for **post-clustering correlation**.

- **Validation Labels**
  - Map incident zones (e.g., E6 highway collapse, Hudiksvall railway failures).

#### Step 1 — Input Preparation

- **Preprocess InSAR + DEM**
  - Apply Hann smoothing or LOESS to InSAR time series.
  - Normalize each input (Asc, Dsc, DEM).
  - Stack:
    - Channel 1: Asc LOS
    - Channel 2: Dsc LOS
    - Channel 3: DEM

- **Synchronize Time Series**
  - Resample all series to a consistent temporal resolution (e.g., weekly or biweekly).

#### Step 2 — Model Design & Training

- **Model Architecture**
  - ConvLSTM encoder over 2D spatial patches × time (3 input channels).
  - LSTM decoder to reconstruct InSAR displacements.
  - Train using **random time permutation trick** (to force temporal pattern learning).

- **Loss Function**
  - Soft-DTW between smoothed InSAR truth and reconstruction.

- **Unsupervised Training**
  - No label leakage.
  - Train only on unlabeled PS points, minimizing soft-DTW reconstruction error.

#### Step 3 — Anomaly Detection & Clustering

- **Anomaly Score**
  - Compute soft-DTW loss per pixel = anomaly strength.
  - Define thresholds (e.g., top 5%) for anomaly flags.

- **Cluster Anomalous Time Series**
  - Use soft-DTW-based clustering or time-series K-Means.
  - Visualize cluster centers (e.g., trend vs. step anomalies).

#### Step 4 — Hydroclimatic & NDVI Correlation (Post-hoc)

- **Correlate Cluster Memberships with Environmental Factors**
  - Rainfall (current + lagged)
  - SWE / snowmelt
  - NDWI / NDVI:
    - NDVI(t): Raw vegetation index
    - ΔNDVI(t): NDVI(t) – NDVI(t–1)
    - Rolling mean of ΔNDVI
    - NDVI trend over time
  - Temperature

- **Analysis Techniques**
  - Correlation plots
  - Cross-correlation functions
  - Lagged regressions

- **Interpretation**
  - Are certain anomaly types driven by climate or land cover?
  - Do **incident zones** align with distinct environmental triggers?

#### Step 5 — Validation & Communication

- **Compare to Incident Labels**
  - Do failed zones cluster differently?
  - Do their time series differ from high-motion but stable zones?

- **Visual Outputs**
  - Anomaly score maps
  - Cluster overlays
  - Environmental overlays

- **Narrative**
  - Even when EGMS maps look similar, motion **dynamics may differ** — a key early warning insight.

---

### Idea #3: Combine Kuzu et al. [4] + Bai et al. [5] — KCC + LSTM Pipeline on EGMS

This is the preferred implementation for vector-based EGMS PS point data.

#### 1. KCC clustering to get deformation behavior categories

- **K-shape clustering** groups time series based on deformation **shape**, not magnitude:
  - Cluster A: linear trend
  - Cluster B: step-like shift
  - Cluster C: seasonal signal
  - Cluster D: flat/stable

- Each cluster has an interpretable mean/centroid time series.
- Spatial linkage (e.g., DBSCAN) refines these into coherent deformation zones.

#### 2. Train an LSTM per cluster

- Each LSTM learns **typical deformation behavior** for its cluster.
- For each time series, compute **prediction or reconstruction error**.
- Anomalies are points that **don’t behave like their cluster peers**.

#### 3. Anomaly types emerge from mismatches

| Cluster Type | Expected Pattern | Anomaly Detected | Interpretation |
|--------------|------------------|------------------|----------------|
| Trend        | Steady rate      | Sudden jump      | Step anomaly in trend cluster |
| Step         | 1-time jump      | Creeping trend   | Trend anomaly in step cluster |
| Flat         | Stable           | Slow subsidence  | Trend anomaly in stable zone |
| Seasonal     | Oscillating      | Monotonic drop   | Failure-like motion in seasonal background |

So even though a failed road segment may **belong** to a step-like cluster, it could still be anomalous due to:
- Larger amplitude
- Early onset
- Compound behavior (e.g., step + acceleration)

#### 4. Cluster the anomalies themselves

Once anomalies are flagged (e.g., top 5–10% by LSTM error), cluster them separately:
- Use:
  - K-shape
  - Time-series K-means
  - DTW-based hierarchical clustering
  - Or t-SNE → DBSCAN for nonlinear shape groups

- Label anomaly types:
  - Cluster 1 = sharp failure
  - Cluster 2 = slow creep
  - Cluster 3 = pre-failure fluctuation
  - Cluster 4 = late-onset drop

#### Summary Table

| Stage              | What’s Clustered              | Purpose                                      |
|-------------------|-------------------------------|----------------------------------------------|
| 1. KCC            | All time series                | Assign expected deformation types            |
| 2. LSTM           | Per-cluster behavior deviation | Identify anomalies within each class         |
| 3. Anomaly Clustering | Only high-error sequences     | Taxonomize failure subtypes for interpretation |

#### Interpretation
This approach gives both:
- **Behavior classification** (what kind of motion is typical here)
- **Anomaly classification** (how this motion is atypical)

It supports:
- **Hazard reporting**
- **Infrastructure screening**
- **Precise early warning rule construction**

---

## 5. Detailed Workflow (for Idea #3)

### Step 1: Preprocess EGMS shapefiles
- Load Asc and Dsc PS data.
- Extract time series as numpy arrays.
- Optionally merge Asc & Dsc into 2-channel time series.
- Filter based on coherence, RMSE, amplitude dispersion.

### Step 2: Apply KCC clustering
- Normalize time series.
- Run K-shape to get shape-based clusters.
- Apply spatial clustering (DBSCAN, distance linkage).
- Label points with cluster ID (behavior type).

### Step 3: Train LSTM per cluster
- Train an LSTM model using typical points in each cluster.
- Compute reconstruction/prediction error for each point.
- Flag anomalies based on high error.

### Step 4: Cluster the anomalies
- Re-normalize high-error sequences.
- Apply DTW-based clustering or similar.
- Interpret clusters as **failure subtypes**.

### Step 5: Post-analysis
- Compare anomaly clusters to:
  - Rainfall
  - SWE / snowmelt
  - NDVI(t), ΔNDVI(t)
  - Slope, geology

- Use:
  - Correlation plots
  - Cross-correlation functions
  - Lag regressions
  - Incident overlays

---

## 6. References

[1] Chang, L., Dollevoet, R. P. B. J., & Hanssen, R. F. (2017). Nationwide Railway Monitoring Using Satellite SAR Interferometry. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 10(2), 596–604. https://doi.org/10.1109/JSTARS.2016.2584783

[2] Schlögl, M., Widhalm, B., & Avian, M. (2021). Comprehensive time-series analysis of bridge deformation using differential satellite radar interferometry based on Sentinel-1. *ISPRS J. Photogramm. Remote Sens.*, 172, 132–146. https://doi.org/10.1016/j.isprsjprs.2020.12.001

[3] Prasanthi, L., Krishnan, S. B., Prasad, K. V., & Chakrabarti, P. (2025). A Deep Embedded Clustering Approach for Detecting Trend Class using Time-Series Sensor Data. *Knowl.-Based Syst.*, 113609. https://doi.org/10.1016/j.knosys.2025.113609

[4] Kuzu, R. S., et al. (2023). Automatic Detection of Building Displacements Through Unsupervised Learning From InSAR Data. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 16, 6931–6947. https://doi.org/10.1109/JSTARS.2023.3297267

[5] Bai, Z., Shen, C., Wang, Y., Lin, Y., Li, Y., & Shen, W. (2025). Bridge Deformation Prediction Using KCC-LSTM With InSAR Time Series Data. *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, 18, 9582–9592. https://doi.org/10.1109/JSTARS.2025.3552665
