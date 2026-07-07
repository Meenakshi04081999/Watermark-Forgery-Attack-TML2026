# Reproduce the Best Leaderboard Result

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Place the script in the dataset root

The script uses relative paths, so it must sit alongside the data:

```
./
├── task_template_alpha1.py
├── clean_targets/            # 1.png ... 200.png
└── watermarked_sources/
    └── WM_1/ ... WM_8/        # 25 images each
```

## 3. Set your API key

In `task_template_alpha1.py`, set `API_KEY` to your leaderboard key.

## 4. Run

```bash
python task_template_alpha1.py
```

This builds `submission.zip` and submits it to the leaderboard.
