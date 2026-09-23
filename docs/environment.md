# Development environment

Every member of the team runs the same Python version and the same pinned
package versions. An experiment that cannot be reproduced on another machine is
not a result.

---

## 1. Python 3.12 — not 3.11, not 3.13

Install **Python 3.12** from <https://www.python.org/downloads/release/python-31210/>.
The file you want is **"Windows installer (64-bit)"**, named `python-3.12.10-amd64.exe`.

Do **not** download a `.tar.xz` file. That is the source code for Linux and has
to be compiled. Do not download the "embeddable package" `.zip` either; it has
no `pip` and cannot create virtual environments.

You can keep other Python versions installed alongside this one. They do not
conflict.

During the installer, tick **"Add python.exe to PATH"**.

### Why this version, and not the newest

Two constraints have to hold at once.

**`causalml` ships compiled extension modules**, so it only installs cleanly on
a Python version it publishes a prebuilt Windows wheel for. On a version with no
wheel, `pip` falls back to building from source, which needs Microsoft C++ Build
Tools and frequently fails.

**python.org stops publishing Windows installers** for a release line long
before that line stops receiving security fixes. Later versions exist as source
code only.

Checked against PyPI and python.org on 20 September 2026:

| Python | `causalml` Windows wheel | Newest Windows installer |
|---|---|---|
| 3.10 | no | — |
| 3.11 | yes | 3.11.9, April 2024 |
| **3.12** | **yes** | **3.12.10, April 2025** |
| 3.13 | no | current |

3.12 is the only line that satisfies both. 3.13 would mean compiling `causalml`
from source. 3.11 would mean running an installer that is more than a year older
than the 3.12 one for no benefit.

The full dependency set in `requirements.txt` was installed and exercised on
both 3.11 and 3.12. Both work. 3.12 is chosen because of the installer age, not
because 3.11 fails.

Verify before continuing:

```
py -3.12 --version
```

Expected output: `Python 3.12.x`

If that prints a version, you already have 3.12 and do not need to install
anything. Any 3.12.x patch level is fine; the pinned package versions are what
make the environment identical across machines, not the Python patch number.

---

## 2. Create the virtual environment

From the repository root:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Your prompt should now start with `(.venv)`. If it does not, the environment is
not active and everything below will install into the wrong place.

`.venv` is in `.gitignore`. It is never committed.

**On macOS or Linux** the two middle commands are:

```
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install the dependencies

```
pip install -r requirements-dev.txt
```

That file includes `requirements.txt`, so this single command installs both the
runtime and the development tooling.

Expect this to take several minutes and download roughly 1 GB.

---

## 4. Create your `.env`

```
copy .env.example .env
```

Open `.env` and adjust anything that differs on your machine. The defaults work
as they are.

`.env` is in `.gitignore` and must never be committed. If you ever find yourself
about to commit it, stop and ask in `#help`.

---

## 5. Verify

```
python scripts/verify_environment.py
```

This checks the Python version, that you are inside a virtual environment, that
every package pinned in `requirements.txt` is installed at exactly that version,
that the script itself covers every pin, that MLflow can write to its tracking
store, and that a two-model uplift estimator actually fits and scores on
synthetic data.

All checks must pass. The script prints the count itself. If that number ever
changes it is because a dependency was added or removed, not because something
is wrong.

The `pin coverage` check exists so that adding a package to `requirements.txt`
and forgetting to add it to the script fails the run, instead of leaving it
silently unverified. `requirements-dev.txt` is deliberately not checked here:
those tools are verified by being used, since CI runs `ruff`, `mypy` and
`pytest` on every pull request.

Paste the full output into the environment task on GitHub. That output is the
evidence that your machine matches everyone else's.

If anything fails, post the whole output in `#help`. Do not start your task on a
broken environment and do not work around a failure by installing a different
version.

---

## 6. MLflow

MLflow 3.x **refuses the plain `./mlruns` directory store**. Running against it
raises:

> The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and
> will not receive further updates.

The project therefore uses a SQLite backend, already set in `.env.example`:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

To open the UI:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then visit <http://127.0.0.1:5000>.

`mlflow.db` and `mlartifacts/` are in `.gitignore`. Experiment results are
shared by committing the code and the config that produced them, never by
committing the tracking database.

---

## Changing a dependency

Pins are not decoration. To add or upgrade a package:

1. Install it in your virtual environment.
2. Run `python scripts/verify_environment.py` and confirm it still passes.
3. Run `pip check` and confirm it reports no broken requirements.
4. Add the exact pinned version to `requirements.txt`.
5. Open a pull request explaining what the package is for.

Never commit the output of `pip freeze` wholesale. It captures transitive
dependencies that nobody chose and makes the file impossible to review.

---

## What is deliberately not installed yet

| Package | Needed from |
|---|---|
| PyTorch | milestone 5, for TARNet and Dragonnet |
| FastAPI, Uvicorn | milestone 5, for the serving API |
| Streamlit | milestone 5, for the dashboard |
| azure-ai-ml | milestone 6, for Azure deployment |

PyTorch alone is over 2 GB. Installing it in week 6 for a model the team will
not train until week 15 wastes everyone's disk and download time.

---

## Verified configuration

Resolved and tested together on Python 3.12, 20 September 2026.
The same set was also verified on Python 3.11.
`pip check` reports no broken requirements with all of the following installed
in one environment:

```
numpy 2.4.6          pandas 3.0.6         scipy 1.17.1
scikit-learn 1.9.1   scikit-uplift 0.5.1  causalml 0.17.0
statsmodels 0.15.0   lightgbm 4.7.0       xgboost 3.2.0
shap 0.51.0          mlflow 3.16.1        optuna 5.0.0
matplotlib 3.11.2    seaborn 0.13.2       PyYAML 6.0.3
python-dotenv 1.2.3  ipykernel 7.3.0
```

`scikit-uplift` and `causalml` coexist in a single environment. They do not
need separate virtual environments.
