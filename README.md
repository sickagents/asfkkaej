# ROGII Wellbore Geology Prediction - 5 VPS Pipeline

## Workflow

```
5 VPS (development) ──> Best params ──> Kaggle Notebook (submit)
```

1. **5 VPS**: Tuning hyperparameter & feature engineering (160 core each)
2. **Ensemble**: Gabungkan hasil terbaik
3. **Kaggle Notebook**: Upload model terbaik ke Kaggle notebook (Code Competition)

## Setup

```bash
git clone https://github.com/sickagents/asfkkaej.git
cd asfkkaej
nano config.json   # ganti vps_id: 1-5
bash loop.sh
```

## Config

```json
{
    "vps_id": 1,                    // GANTI per VPS: 1,2,3,4,5
    "total_vps": 5,
    "competition": "rogii-wellbore-geology-prediction",
    "target": "TVT",
    "optuna_trials": 500,
    "cv_folds": 5,
    "max_session_minutes": 30,
    "seed_base": 42
}
```

## Setelah Semua VPS Selesai

```bash
cd asfkkaej
git pull
python ensemble.py
```

## Catatan Penting: Code Competition

ROGII adalah **Code Competition**:
- Submit **Kaggle Notebook**, bukan CSV
- Notebook jalan di server Kaggle (terbatas: ~9 jam, 16GB RAM)
- 160 core kamu = buat **development & tuning** aja
- Setelah dapat best params, buat notebook di Kaggle dan submit

## Files

```
├── config.json           # Edit vps_id per VPS
├── run.py                # Pipeline utama
├── features.py           # Feature engineering
├── ensemble.py           # Gabungkan hasil
├── loop.sh               # Auto-relog loop
└── data/                 # (auto-download, git-ignored)
```
