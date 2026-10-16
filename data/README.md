# Dataset

The raw CSV files (`train.csv`, `test.csv`) are not committed to this
repository because they exceed GitHub's recommended size limits and are
publicly available.

## Source

**Airline Passenger Satisfaction**  
Kaggle dataset: <https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction>

## Download Instructions

1. Create a free Kaggle account and accept the dataset licence.
2. Download `train.csv` and `test.csv` from the link above.
3. Place both files in this `dataset/` directory:
   ```
   dataset/train.csv
   dataset/test.csv
   ```
4. Run the project pipeline as described in the root `README.md`.

## Expected Schema

| Column | Type | Description |
|--------|------|-------------|
| Unnamed: 0 | int | Row index (excluded from features) |
| id | int | Passenger ID (excluded from features) |
| Gender | str | Male / Female |
| Customer Type | str | Loyal Customer / disloyal Customer |
| Age | int | Passenger age |
| Type of Travel | str | Business travel / Personal Travel |
| Class | str | Business / Eco / Eco Plus |
| Flight Distance | int | Miles |
| Inflight wifi service | int | Rating 0–5 |
| Departure/Arrival time convenient | int | Rating 0–5 |
| Ease of Online booking | int | Rating 0–5 |
| Gate location | int | Rating 0–5 |
| Food and drink | int | Rating 0–5 |
| Online boarding | int | Rating 0–5 |
| Seat comfort | int | Rating 0–5 |
| Inflight entertainment | int | Rating 0–5 |
| On-board service | int | Rating 0–5 |
| Leg room service | int | Rating 0–5 |
| Baggage handling | int | Rating 0–5 |
| Checkin service | int | Rating 0–5 |
| Inflight service | int | Rating 0–5 |
| Cleanliness | int | Rating 0–5 |
| Departure Delay in Minutes | int | Minutes |
| Arrival Delay in Minutes | float | Minutes (contains missing values) |
| satisfaction | str | **Target**: `satisfied` / `neutral or dissatisfied` |
