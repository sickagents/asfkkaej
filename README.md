# Kaggle F1 Pit Stops - 5 VPS Distributed Pipeline

## Setup Sekali (di laptop kamu)

### 1. Buat repo GitHub (private)
```
https://github.com/new
Name: kaggle-f1-pitstops
Visibility: Private
```

### 2. Clone & push code ke GitHub
```bash
cd ~/kaggle-f1-pitstops
git init
git add .
git commit -m "init"
git remote add origin https://github.com/YOUR_USER/kaggle-f1-pitstops.git
git push -u origin main
```

### 3. Buat Kaggle API token
```
https://www.kaggle.com/settings -> API -> Create New Token
Download kaggle.json
```

---

## Setup di Setiap VPS (cuma 1x)

```bash
# Clone repo
git clone https://github.com/YOUR_USER/kaggle-f1-pitstops.git
cd kaggle-f1-pitstops

# Setup Kaggle auth
mkdir -p ~/.kaggle
# Upload kaggle.json ke VPS, lalu:
cp ~/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Setup git auth (supaya bisa push)
git config credential.helper store
# Push sekali biar tersimpan:
git push  # masukkan username + token
```

---

## Edit config.json (PER VPS - BAGIAN INI YANG DIEDIT)

```json
{
    "vps_id": 1,        // <-- GANTI: 1, 2, 3, 4, atau 5
    "total_vps": 5,
    "competition": "playground-series-s6e5",
    "target": "PitNextLap",
    "optuna_trials": 500,
    "cv_folds": 10,
    "max_session_minutes": 50,
    "seed_base": 42
}
```

**Yang perlu diedit per VPS: CUMA `vps_id`**
- VPS 1: `"vps_id": 1`
- VPS 2: `"vps_id": 2`
- VPS 3: `"vps_id": 3`
- VPS 4: `"vps_id": 4`
- VPS 5: `"vps_id": 5`

```bash
# Contoh edit di VPS 3:
# Buka config.json, ganti vps_id jadi 3
nano config.json
```

---

## Jalankan (di setiap VPS)

```bash
cd kaggle-f1-pitstops

# Option 1: Run sekali (kalau yakin selesai dalam 1 jam)
python run.py

# Option 2: Auto-relog loop (kalau butuh beberapa session)
bash loop.sh
```

---

## Ensemble & Submit (di VPS 1 setelah semua selesai)

```bash
cd kaggle-f1-pitstops
git pull                        # tarik hasil dari VPS lain
python ensemble.py              # gabung + submit
```

---

## File Structure
```
kaggle-f1-pitstops/
├── config.json           # EDIT INI SAJA per VPS
├── run.py                # Pipeline utama
├── features.py           # Feature engineering
├── ensemble.py           # Gabungkan hasil
├── loop.sh               # Auto-relog loop
├── .gitignore
├── data/                 # (auto-download, git-ignored)
├── checkpoints/          # (auto-save, di-commit)
├── results/              # (submissions, di-commit)
└── models/               # (git-ignored)
```

## How It Works

```
VPS 1 (vps_id=1) ──git push──> GitHub
VPS 2 (vps_id=2) ──git push──> GitHub
VPS 3 (vps_id=3) ──git push──> GitHub
VPS 4 (vps_id=4) ──git push──> GitHub
VPS 5 (vps_id=5) ──git push──> GitHub
                              │
VPS 1 <───────git pull────────┘
   └── ensemble.py ──> Kaggle submission
```

Each VPS:
- Different vps_id = different random seed
- Pushes checkpoints/results to GitHub
- Can resume if session interrupted

Master (VPS 1):
- Pulls all results
- Optimizes ensemble weights
- Submits to Kaggle
