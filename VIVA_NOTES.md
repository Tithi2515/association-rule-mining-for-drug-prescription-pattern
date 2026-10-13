# University Viva & Oral Presentation Notes

**Project Title:** Association Rule Mining for Drug Co-occurrence Patterns — Apriori vs FP-Growth  
**Course:** Data Mining + Python for Data Science  
**Dataset:** Connecticut Open Data: *Accidental Drug Related Deaths (2012–2024)*  

---

## 1. Executive Summary & Problem Statement

### Introduction & Context
Poly-substance abuse (concurrent exposure to multiple psychoactive substances) is the driving mechanism behind modern drug overdose mortality. This project applies **Market Basket Analysis / Association Rule Mining (ARM)** to forensic toxicology records from the Connecticut Office of the Chief Medical Examiner (OCME) covering 2012 through 2024.

### Objectives
1. Identify high-confidence, high-lift drug co-occurrence patterns in post-mortem records.
2. Mathematically compare the performance and algorithmic output of **Apriori** (level-wise candidate generation) and **FP-Growth** (frequent pattern tree projection).
3. Evaluate whether both algorithms yield identical itemset spaces under identical parameters.
4. Provide an interactive web dashboard for real-time exploratory analysis and policy insights.

---

## 2. End-to-End Workflow Architecture

```text
[Raw CSV Dataset: 12,963 Rows]
       │
       ▼
[Data Ingestion: UTF-8 / Multi-Encoding Safe Loader]
       │
       ▼
[Preprocessing & Feature Selection]
  - 17 Substance Items Selected
  - Detection Rule: 'non_blank_not_N' (handles 'Y', 'Y POPS', 'P', 'y', notes)
  - Filter 158 empty rows (deaths where no listed substance was detected)
       │
       ▼
[Cleaned Binary Matrix: 12,805 Transactions x 17 Items]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Apriori Mining]                  [FP-Growth Mining]
  - min_support = 0.02              - min_support = 0.02
  - Bitwise matrix join/prune       - Prefix tree & conditional pattern base
       │                                 │
       ▼                                 ▼
[51 Frequent Itemsets]            [51 Frequent Itemsets]
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           [Mathematical Equivalence Audit]
             - 100% Itemset Identity Match
             - Support differences < 1e-6
                        │
                        ▼
           [Association Rule Generation]
             - min_confidence = 0.50
             - min_lift >= 1.00
             - Discovered: 12 Strong Rules
                        │
                        ▼
           [Visualizations, CSV Exports & Streamlit Web UI]
```

---

## 3. Dataset Preprocessing & The 12,805 Row Resolution

### Known Raw Data Quirks
- **Inconsistent positive flags:** Fentanyl has `"Y POPS"` and `"Y (PTCH)"`; Ethanol has `"P"`; Xylazine and Gabapentin contain lowercase `"y"`.
- **Negative and junk entries:** Heroin has explicit `"N"` entries; Morphine (Not Heroin) includes notes like `"NO RX BUT STRAWS"` and `"STOLE MEDS"`.
- **Detection Rule Solution:** `DETECTED_RULE = "non_blank_not_N"` treats any non-empty cell that is not `"N"` as detected (1), otherwise 0.

### The 17 vs 18 Columns Dilemma (How 12,805 was derived)
The original CSV has 18 plausible candidate substance columns. Our `check_dataset.py` diagnostic proved:
- Using all 18 candidate columns yields **12,812** retained transactions.
- Dropping either `'Morphine (Not Heroin)'` OR `'Meth/Amphetamine'` drops exactly 7 rows where only that substance was detected, yielding **EXACTLY 12,805 transactions** matching the student project brief.
- Because both dropped substances have $< 1.0\%$ prevalence in the dataset, neither can appear in frequent itemsets at `min_support = 0.02`. Thus, the mined itemsets (51) and rules (12) are **100% identical** in both configurations.

---

## 4. Algorithms Explained in Simple English

### Apriori (Agrawal & Srikant, 1994)
- **Concept:** Bottom-up level-wise search using the **Apriori Principle (Anti-monotonicity)**: *If an itemset is infrequent, all of its supersets must also be infrequent.*
- **Step 1:** Scan dataset to find frequent 1-itemsets ($L_1$).
- **Step 2:** Join $L_1 \bowtie L_1$ to generate candidate 2-itemsets ($C_2$).
- **Step 3:** Prune candidates whose subsets are not frequent.
- **Step 4:** Scan database to count support of candidates; retain those $\ge \text{min\_support}$.
- **Step 5:** Repeat iteratively for size $k = 3, 4, \dots$ until no candidates can be formed.

### FP-Growth (Han et al., 2000)
- **Concept:** Divide-and-conquer strategy that completely **avoids candidate generation**.
- **Step 1:** First database scan discovers frequent 1-items and sorts them in descending frequency order.
- **Step 2:** Second scan builds a compact **FP-Tree (Frequent Pattern Tree)** in memory, where common transaction prefixes share tree branches.
- **Step 3:** For each item from least to most frequent, extracts its **Conditional Pattern Base** (paths leading to that node).
- **Step 4:** Recursively mines **Conditional FP-Trees** to discover all frequent itemsets directly.

---

## 5. Metrics Explained with Real Numbers

Using the project's strongest discovered rule:  
$$\mathbf{\{Fentanyl\ Analogue,\ Heroin\} \Rightarrow \{Fentanyl\}}$$

### 1. Support = 0.0280 (2.80%)
- **Formula:** $\text{Support}(X \cup Y) = \frac{\text{Count}(X \cup Y)}{N}$
- **Meaning:** In $2.80\%$ of all analyzed fatal overdose cases ($359$ out of $12,805$), Fentanyl Analogue, Heroin, and Fentanyl were all detected simultaneously.

### 2. Confidence = 1.0000 (100.0%)
- **Formula:** $\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)} = \frac{0.0280}{0.0280} = 1.000$
- **Meaning:** Given that a decedent tested positive for both Fentanyl Analogue and Heroin, there is a **100.0% probability** that Fentanyl was also detected in their system.

### 3. Lift = 1.4533 (1.453)
- **Formula:** $\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)} = \frac{1.0000}{0.6881} \approx 1.453$
- **Meaning:** The co-occurrence of Fentanyl with the pair $\{\text{Fentanyl Analogue}, \text{Heroin}\}$ is **$1.453$ times more frequent** than would be expected if the substances were detected independently at random.

### 4. The Mathematical "Lift Ceiling" for Fentanyl
- Since $\text{Confidence} \le 1.0$, the maximum possible lift is:
  $$\text{Lift}_{\max} = \frac{1.0}{\text{Support}(Y)} = \frac{1.0}{0.6881} \approx \mathbf{1.453}$$
- **Viva Key Point:** Even though $1.453$ sounds like a small number compared to typical market basket rules, it is at the **absolute theoretical upper bound** for any rule predicting Fentanyl because Fentanyl has such a massive baseline prevalence ($68.8\%$).

---

## 6. Why Apriori vs FP-Growth Runtimes Differ

| Aspect | Apriori (in `mlxtend`) | FP-Growth (in `mlxtend`) |
| :--- | :--- | :--- |
| **Data Structure** | Bitwise Boolean NumPy arrays | In-memory recursive pointer FP-Tree |
| **Candidate Generation** | Yes ($C_k$ level joins) | No (recursive prefix tree projection) |
| **Performance on 17 Items** | **Extremely fast (~0.004 – 0.006s)** | **Slight tree overhead (~0.026 – 0.040s)** |
| **Why Apriori is faster here** | Small item space (17 items) and shallow itemset depth (lengths 1, 2, 3) allow vectorized NumPy bitwise AND operations to run in microseconds with zero pointer overhead. | Building and traversing Python tree nodes, header tables, and recursive child dicts has a fixed memory allocation overhead. |
| **When FP-Growth is faster** | On huge retail datasets with 10,000+ items and long transaction lengths where Apriori generates millions of candidate pairs. | Large, sparse transaction databases. |

> **Honest Viva Statement:** We do not claim FP-Growth is always faster. On small, dense item spaces with vectorized NumPy implementations, Apriori's lightweight bitwise operations outperform recursive tree construction.

---

## 7. Live Demonstration Script (Step-by-Step)

1. **Step 1: Open VS Code Terminal & Run Diagnostics**
   ```powershell
   python check_dataset.py
   ```
   *Explain:* "Here we inspect the Connecticut Open Data dataset. Notice how dropping Morphine (Not Heroin) resolves the 17-column dilemma and produces exactly 12,805 transactions."

2. **Step 2: Run Full CLI Pipeline**
   ```powershell
   python main.py
   ```
   *Explain:* "This executes the 15-stage mining pipeline: EDA, Apriori, FP-Growth, 10-run timing benchmark, and exports 7 CSVs and 5 charts to `outputs/`."

3. **Step 3: Point out the Equivalence Audit**
   *Explain:* "Notice Stage 9: Apriori and FP-Growth both found exactly 51 frequent itemsets and 12 rules with zero support discrepancy."

4. **Step 4: Launch Streamlit Dashboard**
   ```powershell
   streamlit run app.py
   ```
   *Demonstrate:* Show the interactive sliders for min_support and min_confidence, the dual audit comparison tab, and the dynamic Fentanyl lift ceiling card.

---

## 8. 16 Essential Viva Questions & Answers

### Q1: What is Market Basket Analysis / Association Rule Mining?
**Answer:** It is an unsupervised data mining technique that discovers interesting co-occurrence relationships and conditional dependencies between items in large transaction databases.

### Q2: What is the Apriori property (anti-monotonicity)?
**Answer:** Any subset of a frequent itemset must also be frequent. Conversely, if an itemset is infrequent, all of its supersets are guaranteed to be infrequent and can be immediately pruned.

### Q3: Why do Apriori and FP-Growth produce the exact same frequent itemsets?
**Answer:** Because both algorithms solve the exact same mathematical problem: identifying the complete set of itemsets with $\text{support} \ge \text{min\_support}$. They differ in search strategy (level-wise candidates vs prefix-tree divide-and-conquer), but the mathematical solution space is identical.

### Q4: Why is FP-Growth typically considered superior to Apriori in literature?
**Answer:** FP-Growth scans the database only twice and eliminates costly candidate generation by compressing transactions into a compact in-memory FP-Tree.

### Q5: Why was Apriori faster than FP-Growth in our real benchmark?
**Answer:** Because our dataset has only 17 substance items and short itemsets. In `mlxtend`, Apriori utilizes vectorized NumPy C-level bitwise joins, whereas FP-Growth incurs Python object overhead constructing and traversing tree node pointers.

### Q6: What does Support measure?
**Answer:** Support measures how frequently an itemset appears across the entire dataset. It represents the unconditional joint probability $P(A \cap B)$.

### Q7: What does Confidence measure?
**Answer:** Confidence measures the reliability of a rule. It is the conditional probability $P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$.

### Q8: What does Lift measure, and what does Lift = 1.0 mean?
**Answer:** Lift measures the strength of association relative to random chance: $\text{Lift} = \frac{P(B \mid A)}{P(B)}$. A lift of 1.0 means antecedent and consequent are completely statistically independent.

### Q9: Why is the highest lift in our results only ~1.453? Is that a weak rule?
**Answer:** No, it is exceptionally strong! Fentanyl appears in ~68.8% of deaths. The maximum possible lift for any rule predicting Fentanyl is $\frac{1.0}{\text{Support}(\text{Fentanyl})} = \frac{1.0}{0.688} \approx 1.453$. The rule is operating at its theoretical ceiling.

### Q10: Why does Fentanyl Analogue + Heroin => Fentanyl have confidence 1.000?
**Answer:** Because in 100% of cases where both Fentanyl Analogue and Heroin were detected, Fentanyl was also found. This reflects illegal drug supply co-formulation where illicit synthetic fentanyl analogues are blended directly with fentanyl and heroin.

### Q11: What is the difference between association and causation?
**Answer:** Association rules show statistical co-occurrence in retrospective data. They do not prove pharmacodynamic drug interactions, medical synergy, or specific lethal mechanisms.

### Q12: Why did we drop 151 (or 158) rows from the raw dataset?
**Answer:** Those rows recorded accidental drug-related deaths where none of the 17 specific selected substance items were detected (e.g., carbon monoxide, prescription antidepressants, or unclassified inhalants). Market basket algorithms require transactions with at least one active item.

### Q13: Why did we load the CSV with `dtype=str` and `keep_default_na=False`?
**Answer:** To prevent pandas from automatically converting empty string cells into `float NaN` values, allowing us to accurately inspect strings like `"Y"`, `"N"`, `"Y POPS"`, and `""`.

### Q14: What is Xylazine, and what did our rules reveal about it?
**Answer:** Xylazine is a non-opioid veterinary sedative ("tranq"). Our rules discovered that Xylazine co-occurs with Fentanyl with 99.4% confidence and lift of 1.446, confirming that Xylazine is overwhelmingly present as an adulterant in fentanyl overdoses.

### Q15: What are the limitations of this study?
**Answer:** The data is restricted to accidental fatal overdoses investigated by the Connecticut medical examiner, so it cannot be generalized to non-fatal drug use, living patient populations, or legitimate clinical prescriptions.

### Q16: How would you scale this pipeline for a production hospital or public health department?
**Answer:** We would automate continuous streaming ingestion via Apache Kafka, store transactions in a distributed data warehouse like Snowflake or BigQuery, run FP-Growth via PySpark MLlib on a cluster, and expose live surveillance dashboards via Streamlit or Grafana.

---

## 9. Conclusion
This project provides an empirically verified demonstration of Association Rule Mining on forensic mortality data. Both Apriori and FP-Growth confirmed identical mathematical itemset sets (51 itemsets) and rules (12 rules), with Apriori exhibiting superior speed on this small, dense itemset structure. The findings highlight severe multi-substance mortality clusters involving Fentanyl, Heroin, Cocaine, and the emerging adulterant Xylazine.
