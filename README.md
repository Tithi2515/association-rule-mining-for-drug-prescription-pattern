# Association Rule Mining for Drug Prescription Pattern: Apriori vs FP-Growth

A complete, beginner-friendly Python data mining demonstration project for Windows and VS Code. This project applies Market Basket Analysis / Association Rule Mining to Connecticut Accidental Drug-Related Deaths data (2012–2024), comparing the classical **Apriori** algorithm and the prefix-tree-based **FP-Growth** algorithm.

---

## 📌 Critical Domain & Ethics Clarification

> [!IMPORTANT]
> **This dataset records accidental drug-related DEATHS, NOT clinical prescriptions or treatment plans.**
> - Each transaction represents post-mortem toxicology findings for an individual decedent as documented by the Connecticut Office of the Chief Medical Examiner (OCME).
> - **Association rules indicate statistical co-occurrence**, *never* therapeutic drug combinations, physician prescribing practices, or proven pharmacological causation.
> - **The Fentanyl Lift Ceiling:** Fentanyl is detected in ~68.8% of retained death records. Because $\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}}{\text{Support}(B)}$, the mathematical maximum possible lift for any rule predicting Fentanyl is $\frac{1.0}{\text{Support}(\text{Fentanyl})} \approx \frac{1.0}{0.688} \approx \mathbf{1.453}$. Therefore, a modest lift of 1.45 does **not** signify a weak rule; it is near the theoretical mathematical ceiling.
> - **Perfect Confidence (1.000):** The rule $\text{Fentanyl Analogue} + \text{Heroin} \Rightarrow \text{Fentanyl}$ achieves 1.000 confidence because illicit fentanyl analogues are overwhelmingly co-adulterated into the street fentanyl and heroin supply.

---

## 📁 Project Structure

```text
drug_arm_project/
│
├── data/
│   └── accidental_drug_deaths.csv    <- Connecticut Open Data CSV
│
├── src/
│   ├── __init__.py                   <- Package initializer
│   ├── data_loader.py                <- Multi-encoding CSV reader with whitespace strip
│   ├── preprocessing.py              <- Binary converter, empty row filter
│   ├── eda.py                        <- Substance frequencies, percentages, top 10
│   ├── mining.py                     <- Apriori, FP-Growth, rule generation, timing
│   └── plots.py                      <- High-resolution Matplotlib visual charts
│
├── outputs/                          <- Automatically generated CSVs and PNG charts
│   ├── cleaned_transactions.csv
│   ├── frequent_itemsets_apriori.csv
│   ├── frequent_itemsets_fpgrowth.csv
│   ├── rules_apriori.csv
│   ├── rules_fpgrowth.csv
│   ├── algorithm_comparison.csv
│   ├── substance_frequency.csv
│   ├── top_10_substances.png
│   ├── support_vs_confidence.png
│   ├── runtime_comparison.png
│   ├── runtime_vs_support.png
│   └── top_rules_by_lift.png
│
├── config.py                         <- Settings, columns, rules, targets
├── check_dataset.py                  <- Column diagnostics & 12,805 target solver
├── main.py                           <- Full 15-stage CLI pipeline
├── app.py                            <- Interactive Streamlit web interface
├── requirements.txt                  <- Python package dependencies
├── README.md                         <- This setup and documentation guide
└── VIVA_NOTES.md                     <- Presentation notes & 15+ Viva Q&A
```

---

## 🚀 Step-by-Step Setup Guide (Windows & VS Code)

Follow these numbered steps to run the project from scratch on Windows.

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File** > **Open Folder...** (or press `Ctrl + K`, `Ctrl + O`).
3. Select this project folder (`PDS-DM Project` or `drug_arm_project`).
4. Open the integrated terminal in VS Code:
   - Go to menu: **Terminal** > **New Terminal** (or press ``Ctrl + ` ``).

---

### Step 2: Create and Activate a Python Virtual Environment

In the VS Code terminal (PowerShell), run:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

> [!TIP]
> **PowerShell Execution Policy Error Fix:**
> If you see `File ... activate.ps1 cannot be loaded because running scripts is disabled on this system`, run this command in your PowerShell terminal and retry activation:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .venv\Scripts\activate
> ```
> When activated, you will see `(.venv)` at the beginning of your command line.

---

### Step 3: Install Required Packages

Install all necessary libraries using `pip`:

```powershell
pip install -r requirements.txt
```

This installs:
- `pandas` (Data manipulation)
- `numpy` (Array operations)
- `mlxtend` (Apriori, FP-Growth, and Association Rules)
- `matplotlib` (Plotting and chart exports)
- `streamlit` (Interactive web application)

---

### Step 4: Verify the Dataset Location

Make sure the dataset is located at:
```text
data/accidental_drug_deaths.csv
```

If it is not already present, download it directly:
- **Direct Download URL:** [https://data.ct.gov/api/views/rybz-nyjw/rows.csv?accessType=DOWNLOAD](https://data.ct.gov/api/views/rybz-nyjw/rows.csv?accessType=DOWNLOAD)
- Save the file as `accidental_drug_deaths.csv` inside the `data/` folder.

---

### Step 5: Run the Dataset Diagnostics (`check_dataset.py`)

Run the dataset diagnostic script:

```powershell
python check_dataset.py
```

#### What this script does:
1. Inspects all 18 candidate substance columns.
2. Evaluates the two detection rules:
   - `non_blank_not_N`: Checks non-blank values excluding `"N"`.
   - `starts_with_Y`: Checks values starting with uppercase `"Y"`.
3. Tests dropping each column individually to see how many rows are kept.
4. Identifies which columns produce **exactly 12,805 transactions** (marked with `<== equals report`).

**Result:**
- Keeping all 18 candidate columns retains **12,812 transactions**.
- Dropping either `'Morphine (Not Heroin)'` OR `'Meth/Amphetamine'` retains **exactly 12,805 transactions**.
- At `min_support = 0.02`, neither of those small columns appears in frequent itemsets anyway, so frequent itemsets (51) and rules (12) are **100% identical**!

---

### Step 6: Configure Substance Columns in `config.py` (Optional)

Open `config.py` in VS Code.

- By default, `SUBSTANCE_COLUMNS` contains all 18 candidates. `main.py` will display a friendly warning if 18 columns are used, and will run smoothly.
- To match the report's exact 17 columns and 12,805 transactions, comment out or remove `'Morphine (Not Heroin)'` from `SUBSTANCE_COLUMNS` in `config.py`:
  ```python
  SUBSTANCE_COLUMNS = [
      "Heroin",
      "Cocaine",
      "Fentanyl",
      "Fentanyl Analogue",
      "Oxycodone",
      "Oxymorphone",
      "Ethanol",
      "Hydrocodone",
      "Benzodiazepine",
      "Methadone",
      "Meth/Amphetamine",
      "Amphet",
      "Tramad",
      "Hydromorphone",
      # "Morphine (Not Heroin)",  # Excluded to yield exactly 17 items and 12,805 transactions
      "Xylazine",
      "Gabapentin",
      "Opiate NOS",
  ]
  ```

---

### Step 7: Run the Full CLI Pipeline (`main.py`)

Execute the complete 15-stage pipeline:

```powershell
python main.py
```

This runs all data mining steps, generates CSV files, creates charts in `outputs/`, and prints a side-by-side calculated vs reported summary table.

---

### Step 8: Launch the Interactive Streamlit Web App (`app.py`)

Launch the web dashboard using Python's module runner:

```powershell
python -m streamlit run app.py
```
*(Alternatively: `streamlit run app.py` if your Scripts directory is in your Windows PATH).*

> [!NOTE]
> **Node.js / npm Notice:**
> If you previously attempted `npm run dev` or received `ENOENT package.json does not exist`, please note this is a pure Python project utilizing Streamlit, pandas, and mlxtend. It does not use Node.js or `package.json`. Use `python -m streamlit run app.py` instead.

Your web browser will open at `http://localhost:8501`.
- Adjust the **min_support**, **min_confidence**, and **min_lift** sliders.
- Toggle between Apriori and FP-Growth, or run a **Dual Audit**.
- Switch column selection modes (18 candidates vs 17 columns).
- Download generated CSV results using the download buttons.

---

## 📊 Expected Outputs

After running `main.py`, the following files are produced in the `outputs/` folder:

| File Name | Description |
| :--- | :--- |
| `cleaned_transactions.csv` | Cleaned 1/0 binary transaction matrix of deaths |
| `frequent_itemsets_apriori.csv` | 51 frequent itemsets mined via Apriori |
| `frequent_itemsets_fpgrowth.csv` | 51 frequent itemsets mined via FP-Growth |
| `rules_apriori.csv` | 12 association rules discovered via Apriori |
| `rules_fpgrowth.csv` | 12 association rules discovered via FP-Growth |
| `algorithm_comparison.csv` | Side-by-side equivalence and benchmark summary |
| `substance_frequency.csv` | Prevalence counts and percentages for all substances |
| `top_10_substances.png` | Horizontal bar chart of top detected substances |
| `support_vs_confidence.png` | Scatter plot of rules colored by Lift |
| `runtime_comparison.png` | Bar chart of first-run vs median execution times |
| `runtime_vs_support.png` | Scalability plot across support thresholds (0.01 to 0.08) |
| `top_rules_by_lift.png` | Horizontal bar chart ranking the strongest rules by Lift |

---

## 🌐 Cloud Deployment Guide

### Why Vercel Fails for Streamlit Applications
If you attempted to deploy this repository on **Vercel** and received the error:
```text
Found app.py, main.py but none export a top-level "app", "application", or "handler" variable.
```
**Reason:** Vercel's Python runtime (`@vercel/python`) is exclusively designed for **stateless WSGI/ASGI serverless functions** (such as Flask or FastAPI). Streamlit is a **stateful, long-running application** that maintains continuous in-memory session states and persistent **WebSockets** between the browser and server. Serverless function runtimes cannot maintain persistent WebSockets or long-running Python processes.

---

### Option 1: Streamlit Community Cloud (Recommended — 100% Free & Native)

Streamlit Community Cloud is the official hosting platform for Streamlit applications. It deploys directly from your GitHub repository in 1 click:

1. Push your repository to GitHub:
   ```powershell
   git add .
   git commit -m "Configure Streamlit Cloud and Render deployment"
   git push origin main
   ```
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"** (or **"Create app"**).
4. Fill in the deployment form:
   - **Repository:** `Tithi2515/association-rule-mining-for-drug-prescription-pattern`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**
   - Streamlit Cloud automatically detects `requirements.txt` and `.streamlit/config.toml`, installs dependencies, and serves your live web application with a permanent public URL (e.g. `https://your-app.streamlit.app`).

---

### Option 2: Render (Free Cloud Web Service)

This repository includes pre-configured [Procfile](file:///c:/Users/Tithi/OneDrive/Desktop/PDS-DM%20Project/Procfile) and [render.yaml](file:///c:/Users/Tithi/OneDrive/Desktop/PDS-DM%20Project/render.yaml) for 1-click web service hosting:

1. Sign up at **[render.com](https://render.com)**.
2. Click **"New +"** > **"Web Service"**.
3. Connect your GitHub repository `association-rule-mining-for-drug-prescription-pattern`.
4. Configure settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## 🛠️ Troubleshooting & Common Fixes

### 1. `ModuleNotFoundError: No module named 'mlxtend'` or `'streamlit'`
- **Cause:** Your terminal is using the global Python instead of the virtual environment.
- **Fix:** Activate `.venv` first:
  ```powershell
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 2. `FileNotFoundError: Dataset not found at ...`
- **Cause:** `accidental_drug_deaths.csv` is missing from the `data/` directory.
- **Fix:** Ensure a folder named `data` exists in the project root, and place `accidental_drug_deaths.csv` inside it.

### 3. CSV Encoding Errors (`UnicodeDecodeError`)
- **Cause:** Non-standard Windows locale or UTF-8 BOM encoding.
- **Fix:** Our `src/data_loader.py` automatically cascades through `utf-8`, `utf-8-sig`, and `latin-1`.

### 4. `TypeError: association_rules() got an unexpected keyword argument 'num_itemsets'`
- **Cause:** Different versions of `mlxtend` handle `num_itemsets` differently.
- **Fix:** Our `src/mining.py` inspects the function signature via `inspect.signature` and dynamically adapts to any `mlxtend` version.

### 5. `KeyError: The following requested substance columns were not found`
- **Cause:** Typos or trailing spaces in column names in `config.py`.
- **Fix:** Run `python check_dataset.py` to see the exact cleaned column headers present in the CSV.
