We thank all reviewers for their time and feedback. Below, we address common and main review points while the rest are given separately for each reviewer.

Following the reviewers' suggestions, we have included *additional results for natural partitions* (to be added to the main paper). Given the opportunity, we will edit and reorganize some material as suggested by Reviewer 36iB. This is feasible for the camera-ready version.

**New Results:**
Attached extra PDF has results for Vehicle and HHNP using natural partitions (NP). Vehicle's heterogeneity is between NIID1 and NIID2, and the results are similar to those in our work (Fig.1 \& Table 1). HHNP shows much higher heterogeneity across the target and features (Fig.1-2; Table 1), diverging from Vehicle (w/ NP) and NIID1/2 partitions. This indicates that FL setting is more complex, and more thorough studies are needed, where FedImpute can play a crucial role.

**The Choice/Development of Fed Imputation Methods (IMs):**
- zCkk questioned if the provided IMs are SOTA.
- jduB questioned the novelty of IMs.
- 36iB commented on the description of IMs in the main body.

We developed a novel benchmarking tool for fed-imputation evaluation. Developing new fed-IMs is beyond the scope of this work (see [a3,a9,a75], 'a' refers to appendix). Thus, we rely on the following inclusion criteria: 1) IMs are well-known and easily extensible to a federated version, e.g., mean imputation, MissForest, and Expectation Maximization -- the last two are new. 2) Use existing Fed-IMs, e.g., Fed-MIWAE (2023), Fed-GAIN (2021), and Fed Linear-ICE (2020). We also added 2 more SOTA IMs (not-MIWAE (2021) and GNR (2023)). Thus, the provided Fed-IMs are SOTA.

Although we mentioned extending MissForest and EM to the FL setting (*line 240*), we will clarify this in the main paper. Additionally, using different aggregation functions/strategies is a standard way to enhance an FL model without altering other aspects [r1, r2] ('r' citation are in extra PDF). So we provide different aggregation functions to enhance FedImpute’s usability. Note that since all our IMs are federated, we dropped the "Fed" prefix for simplicity (e.g., MissForest instead of Fed MissForest). We will clarify this in the main paper. 

These aspects were considered at the design level to facilitate the integration of future Fed IMs, given that research in this area is still in its early stages.


**Hyperparameter Tuning:**
Tuning can affect NN-based models' performance (as noted in our limitations). But, it doesn't address the fundamental issues caused by the non-IIDness — whether missing or observable — which is well-documented in FL [a33]. Since we focused on exploring differences across missing data scenarios, we avoided tuning, which also saved a lot of time. For generality, a broader set of benchmarks is needed, covering data-partitioning based on different distributions, correlations (with features and targets), missing mechanisms, missing ratios, correlation strengths, and levels of diversity and heterogeneity.


**Datasets:**
We focus on the cross-silo setting for tabular data, where, compared to the cross-device setting, the number of clients is fewer, and data per client is larger. The datasets we used are from other FL studies [r1, r3, r4, a66]. They range from 10k to 20k in size, with feature dimensions from 7 to 41, covering various domains and complexities. Note that most other FL studies (e.g., 36iB[1,2,3]) use textual and image datasets, which compared to tabular data, require different methods of handling missing data.


**Generalizability:**
We established the complexity for fed-imputation evaluation (*ls 96-106*). Despite many simplifications, 56 scenarios were needed for a single feature; the number is much larger in general settings. A comprehensive and generalizable benchmark across all aspects is infeasible for a single study. We expect FedImpute to be useful for future studies.

For serious evaluation, a fixed set of missingness and/or data-partitioning is insufficient. Hence, we report average results over multiple runs (with divergence given by error bars in the appendix). Each round involves creating new partitions for clients using the partitioning strategies (more one this below), generating new missing values with missing mechanisms of the same types, variety, and heterogeneity (which are decided based on the scenario). This approach ensures generalizability beyond a single missingness, partitioning, or dataset. We will clarify this setup in the main paper.

Simulation is a widely accepted approach in FL settings [8, a33, r5, r6], where missing mechanisms are used in missing-data studies and data-partitioning mechanisms are used in FL studies. For generalizability, missing data simulation is crucial as it provides ground truth for evaluating imputation quality (which, i.e., the ground truth, is not available for real data with missing values). Simulation allows the use of the same dataset (with its fixed naturally occurring feature correlations) across different missing mechanisms and data partitions, enhancing Fed-IMs evaluation.


**Data-Partitioning:**
As mentioned, new partitions are created (in each round) with iid or skewed quantities and features/targets at varying levels (NIID1/2), which are widely used strategies [a33]. While natural partitions are important, they remain fixed. Using partitioning strategies allows to explore a broader range of data heterogeneity and its effects on fed-imputation. Dirichlet-based partitioning is common in FL [a33, r1], where skewness is controlled by alpha-parameter and the feature used for partitioning. For consistency, when selecting a feature to be used with Dirichlet-based partitioning, we use the feature with the maximum average pairwise correlation, if the feature is numerical, we discretize it using 20 bins. Although detailed in the appendix, we will add this information to the main paper as well.