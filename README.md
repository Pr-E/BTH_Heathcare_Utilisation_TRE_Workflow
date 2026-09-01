### Active Blackpool / BTH TRE Analytical Workflow

* **Purpose:** Reproducible workflow to support evaluation of the Active Blackpool exercise referral pathway using linked BTH MSK, ED and inpatient data within the TRE.

* **Development approach:** Built and validated using synthetic data before translation to the real BTH datasets.

* **Workflow stages:**
  Data ingestion → data-quality checks → cleaning → preprocessing → linkage → cohort/index construction → healthcare-utilisation outcomes → descriptive analysis → propensity adjustment → comparative pre/post modelling → exploratory clustering.

* **Primary comparative design:**
  Sports-linked BTH pathway compared with the Wider MSK population using:

  * structural positivity checks;
  * common-support assessment;
  * ATT weighting;
  * standardised mean differences;
  * 1:3 propensity-score matching as sensitivity analysis.

* **Outcome modelling:**
  Poisson GEE with a **group × period interaction** and **log person-time offset**, with Negative Binomial GEE as a distributional sensitivity analysis.

* **Clustering:**
  Secondary exploratory K-means analysis using only baseline ED, inpatient and emergency inpatient utilisation rates. Demographic, geographic, pathway and follow-up variables are used only afterwards for cluster profiling.

* **Reproducibility and auditability:**
  Modular TRE-ready structure with stage-specific QA outputs, terminal summaries, audit files and clear next-step instructions.

* **TRE translation:**
  All real-data source semantics, cohort counts, missingness, linkage, overlap, balance, model estimates and clustering solutions must be regenerated and reassessed inside the TRE.

* **Important interpretation:**
  Synthetic results validate the analytical workflow and methodology; they are **not carried forward as substantive Active Blackpool programme findings**.
