# generate_data.py
#
# Deterministic synthetic Spaceship-Titanic data for the E2E track.
# Same schema as the Kaggle competition, no credentials, byte-identical
# output for identical arguments — every E2E run sees the same data.
#
# The target carries real signal (CryoSleep, HomePlanet, cabin side, spend)
# behind noise, missing values, and two spend columns that are pure noise:
# a DummyClassifier sits at ~0.5, honest feature work reaches ~0.8.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOME_PLANETS = ["Earth", "Europa", "Mars"]
DESTINATIONS = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
DECKS = ["A", "B", "C", "D", "E", "F", "G", "T"]
FIRST_NAMES = [
    "Altair", "Berenike", "Cassian", "Deneb", "Elara", "Fomal", "Gliese",
    "Hyadum", "Izar", "Juxta", "Kepler", "Lyra", "Mirach", "Nunki",
]
LAST_NAMES = [
    "Aldervine", "Bristlecone", "Cometwright", "Dustfarer", "Emberlyn",
    "Frostvale", "Gravwell", "Holloway", "Ionsong", "Jettison",
]
MISSING_RATE = 0.02


def _make_frame(rng: np.random.RandomState, n_rows: int, start_group: int) -> pd.DataFrame:
    group_ids = []
    passenger_ids = []
    group = start_group
    while len(passenger_ids) < n_rows:
        size = rng.choice([1, 1, 1, 2, 2, 3])
        for member in range(1, size + 1):
            if len(passenger_ids) == n_rows:
                break
            group_ids.append(group)
            passenger_ids.append(f"{group:04d}_{member:02d}")
        group += 1

    home_planet = rng.choice(HOME_PLANETS, size=n_rows, p=[0.54, 0.25, 0.21])
    cryo_sleep = rng.random_sample(n_rows) < 0.35
    deck = rng.choice(DECKS, size=n_rows)
    cabin_num = rng.randint(0, 1500, size=n_rows)
    side = rng.choice(["P", "S"], size=n_rows)
    cabin = np.array(
        [f"{d}/{n}/{s}" for d, n, s in zip(deck, cabin_num, side)]
    )
    destination = rng.choice(DESTINATIONS, size=n_rows, p=[0.69, 0.21, 0.10])
    age = np.clip(rng.normal(29, 14, size=n_rows), 0, 79).round(0)
    vip = rng.random_sample(n_rows) < 0.023

    def spend(scale: float) -> np.ndarray:
        amounts = rng.lognormal(mean=scale, sigma=1.6, size=n_rows)
        amounts[cryo_sleep] = 0.0
        return amounts.round(1)

    room_service = spend(4.0)
    food_court = spend(4.6)
    shopping_mall = spend(4.2)
    spa = spend(4.4)
    vr_deck = spend(4.3)

    first = rng.choice(FIRST_NAMES, size=n_rows)
    last = rng.choice(LAST_NAMES, size=n_rows)
    name = np.array([f"{f} {l}" for f, l in zip(first, last)])

    # Latent signal, computed on true (pre-masking) values. Spa and VRDeck
    # deliberately carry no weight: two decoy features.
    luxury_spend = room_service + spa
    latent = (
        2.4 * cryo_sleep.astype(float)
        + 0.9 * (home_planet == "Europa").astype(float)
        - 0.5 * (home_planet == "Earth").astype(float)
        + 0.4 * (side == "S").astype(float)
        - 0.9 * np.log1p(luxury_spend) / np.log1p(luxury_spend).max()
        + rng.normal(0, 0.9, size=n_rows)
    )
    transported = latent > np.median(latent)

    frame = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "HomePlanet": home_planet,
            "CryoSleep": cryo_sleep,
            "Cabin": cabin,
            "Destination": destination,
            "Age": age,
            "VIP": vip,
            "RoomService": room_service,
            "FoodCourt": food_court,
            "ShoppingMall": shopping_mall,
            "Spa": spa,
            "VRDeck": vr_deck,
            "Name": name,
            "Transported": transported,
        }
    )

    # Missingness after the target is fixed, never on identifiers or target.
    # Bool feature columns become object first (True/False/NaN), matching
    # how the Kaggle CSVs parse.
    maskable = [
        "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
    ]
    frame["CryoSleep"] = frame["CryoSleep"].astype(object)
    frame["VIP"] = frame["VIP"].astype(object)
    for column in maskable:
        mask = rng.random_sample(n_rows) < MISSING_RATE
        frame.loc[mask, column] = np.nan
    return frame


def generate(*, rows: int, test_rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_df = _make_frame(rng, rows, start_group=1)
    test_df = _make_frame(rng, test_rows, start_group=8000)
    test_df = test_df.drop(columns=["Transported"])
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic Spaceship-Titanic data"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    train_df, test_df = generate(
        rows=args.rows, test_rows=args.test_rows, seed=args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)
    print(
        f"Wrote {len(train_df)} train rows and {len(test_df)} test rows "
        f"to {args.out_dir} (seed {args.seed})"
    )


if __name__ == "__main__":
    main()
