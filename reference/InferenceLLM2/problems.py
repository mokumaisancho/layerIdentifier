"""Problem set loader and GSM8K-style problem definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Problem:
    """A single math reasoning problem."""

    id: str
    question: str
    ground_truth: float
    problem_type: str  # arithmetic, geometry, counting, comparison, multi_step
    expected_error_type: str = "unknown"  # template, structural, algorithmic, unknown
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "problem_type": self.problem_type,
            "expected_error_type": self.expected_error_type,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Problem:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# 50 problems: 10 per type, hand-curated for variety
# These are GSM8K-style problems with known ground truth
BUILTIN_PROBLEMS: list[dict] = [
    # === ARITHMETIC (10) ===
    {"id": "arith_001", "question": "John has 5 apples. He gives away 2. How many does he have?", "ground_truth": 3.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "arith_002", "question": "A store sells pencils for $3 each. If Maria buys 7 pencils and pays with a $50 bill, how much change does she get?", "ground_truth": 29.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "arith_003", "question": "A train travels 60 miles per hour for 3 hours. How far does it go?", "ground_truth": 180.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "arith_004", "question": "A bakery makes 24 muffins per batch. They make 6 batches before noon and 4 batches after noon. How many muffins total?", "ground_truth": 240.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "arith_005", "question": "Tom earns $15 per hour. He worked 8 hours on Monday and 6 hours on Tuesday. He spent $23 on lunch. How much money does he have left?", "ground_truth": 187.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "arith_006", "question": "A rectangle has length 12 cm and width 8 cm. What is its perimeter?", "ground_truth": 40.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "arith_007", "question": "Lisa has 3 times as many stickers as Mark. Mark has 14 stickers. How many stickers do they have together?", "ground_truth": 56.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "arith_008", "question": "A pizza costs $12.50. If 5 friends split it equally, how much does each pay?", "ground_truth": 2.5, "problem_type": "arithmetic", "expected_error_type": "algorithmic", "difficulty": "medium"},
    {"id": "arith_009", "question": "A farmer has 156 eggs. He sells 48 at the market and uses 12 for baking. He then buys 36 more. How many eggs does he have?", "ground_truth": 132.0, "problem_type": "arithmetic", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "arith_010", "question": "A book costs $24. The store offers a 25% discount. What is the sale price?", "ground_truth": 18.0, "problem_type": "arithmetic", "expected_error_type": "algorithmic", "difficulty": "medium"},

    # === GEOMETRY (10) ===
    {"id": "geom_001", "question": "A triangle has a base of 10 cm and height of 6 cm. What is its area?", "ground_truth": 30.0, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "easy"},
    {"id": "geom_002", "question": "A circular garden has a radius of 7 meters. What is its circumference? (Use pi = 3.14)", "ground_truth": 43.96, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "medium"},
    {"id": "geom_003", "question": "Blake runs back and forth across a 40-yard field 15 times. Kelly runs back and forth once, then runs to the 40-yard line and back 34 times. How much farther does the winner run than the loser?", "ground_truth": 80.0, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "geom_004", "question": "A room is 5 meters long, 4 meters wide, and 3 meters high. What is the volume?", "ground_truth": 60.0, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "easy"},
    {"id": "geom_005", "question": "A square has an area of 144 square inches. What is its perimeter?", "ground_truth": 48.0, "problem_type": "geometry", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "geom_006", "question": "A cylinder has radius 3 cm and height 10 cm. What is its volume? (Use pi = 3.14)", "ground_truth": 282.6, "problem_type": "geometry", "expected_error_type": "algorithmic", "difficulty": "medium"},
    {"id": "geom_007", "question": "A rectangular pool is 20 meters long and 10 meters wide. A path of width 2 meters surrounds it. What is the area of the path?", "ground_truth": 128.0, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "geom_008", "question": "The angles in a triangle sum to 180 degrees. If two angles are 45 and 65 degrees, what is the third angle?", "ground_truth": 70.0, "problem_type": "geometry", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "geom_009", "question": "A sphere has radius 5 cm. What is its surface area? (Use pi = 3.14)", "ground_truth": 314.0, "problem_type": "geometry", "expected_error_type": "algorithmic", "difficulty": "medium"},
    {"id": "geom_010", "question": "A trapezoid has bases of 8 cm and 12 cm and height of 5 cm. What is its area?", "ground_truth": 50.0, "problem_type": "geometry", "expected_error_type": "structural", "difficulty": "medium"},

    # === COUNTING (10) ===
    {"id": "count_001", "question": "How many ways can you arrange 4 books on a shelf?", "ground_truth": 24.0, "problem_type": "counting", "expected_error_type": "structural", "difficulty": "easy"},
    {"id": "count_002", "question": "In a class of 30 students, 18 like math and 15 like science. If 8 like both, how many like neither?", "ground_truth": 5.0, "problem_type": "counting", "expected_error_type": "structural", "difficulty": "medium"},
    {"id": "count_003", "question": "A coin is flipped 3 times. How many possible outcomes are there?", "ground_truth": 8.0, "problem_type": "counting", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "count_004", "question": "How many diagonals does a hexagon have?", "ground_truth": 9.0, "problem_type": "counting", "expected_error_type": "algorithmic", "difficulty": "medium"},
    {"id": "count_005", "question": "A restaurant offers 3 appetizers, 5 main courses, and 2 desserts. How many different 3-course meals can you order?", "ground_truth": 30.0, "problem_type": "counting", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "count_006", "question": "From a group of 7 people, how many ways can you choose a committee of 3?", "ground_truth": 35.0, "problem_type": "counting", "expected_error_type": "algorithmic", "difficulty": "medium"},
    {"id": "count_007", "question": "How many integers between 1 and 100 are divisible by both 3 and 5?", "ground_truth": 6.0, "problem_type": "counting", "expected_error_type": "structural", "difficulty": "medium"},
    {"id": "count_008", "question": "A password must be 4 digits. How many possible passwords are there?", "ground_truth": 10000.0, "problem_type": "counting", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "count_009", "question": "In a bag of 20 marbles, 5 are red, 8 are blue, and 7 are green. How many ways can you pick 2 marbles of the same color?", "ground_truth": 53.0, "problem_type": "counting", "expected_error_type": "algorithmic", "difficulty": "hard"},
    {"id": "count_010", "question": "How many ways can 6 people sit around a circular table?", "ground_truth": 120.0, "problem_type": "counting", "expected_error_type": "algorithmic", "difficulty": "medium"},

    # === COMPARISON (10) ===
    {"id": "comp_001", "question": "Alice is 12 years old. Bob is 3 years older than Alice. Carol is twice as old as Bob was 5 years ago. Who is the oldest?", "ground_truth": 20.0, "problem_type": "comparison", "expected_error_type": "structural", "difficulty": "medium"},
    {"id": "comp_002", "question": "Store A sells a shirt for $25 with 20% off. Store B sells the same shirt for $22 with no discount. Which store is cheaper and by how much?", "ground_truth": 2.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "comp_003", "question": "A car travels 120 miles in 2 hours. A bus travels 180 miles in 3 hours. Which is faster?", "ground_truth": 0.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "comp_004", "question": "Tom saves $50 per week. Lisa saves $75 per week but started 3 weeks later than Tom. After how many weeks will Lisa have saved more than Tom?", "ground_truth": 9.0, "problem_type": "comparison", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "comp_005", "question": "Team A scored 45, 52, and 38 points in three games. Team B scored 41, 55, and 42 points. Which team has the higher average?", "ground_truth": 46.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "comp_006", "question": "A 15% tip on a $60 meal versus a $12 flat service charge. Which costs more and by how much?", "ground_truth": 3.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "comp_007", "question": "Rectangle A is 8 by 12. Rectangle B is 9 by 10. Which has a larger area and by how much?", "ground_truth": 6.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "comp_008", "question": "Investment A grows 5% per year for 3 years on $1000. Investment B grows 8% once on $1000. Which yields more?", "ground_truth": 57.62, "problem_type": "comparison", "expected_error_type": "algorithmic", "difficulty": "hard"},
    {"id": "comp_009", "question": "Newton has 11 apples and gives 6 to his friend. How many apples does Newton have left?", "ground_truth": 5.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "easy"},
    {"id": "comp_010", "question": "A movie ticket costs $12 for adults and $8 for children. A family of 2 adults and 3 children goes to the movies with $50. How much change do they get?", "ground_truth": 2.0, "problem_type": "comparison", "expected_error_type": "template", "difficulty": "easy"},

    # === MULTI-STEP (10) ===
    {"id": "multi_001", "question": "A factory produces 500 units per day. After a machine upgrade, production increases by 20%. After 10 days, 15% of the units are defective and discarded. How many good units were produced?", "ground_truth": 4250.0, "problem_type": "multi_step", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "multi_002", "question": "Sarah buys 3 notebooks at $4 each and 2 pens at $2 each. She pays with a $20 bill. The cashier gives her a 10% loyalty discount on the total. How much change does she get?", "ground_truth": 10.6, "problem_type": "multi_step", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "multi_003", "question": "A pool holds 10000 liters. A hose fills it at 500 liters per hour. After 8 hours, a drain opens that drains 200 liters per hour. How long until the pool is full?", "ground_truth": 28.0, "problem_type": "multi_step", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "multi_004", "question": "A store buys shirts for $15 each and sells them for $25. They sell 80% of their 200 shirts at full price and the rest at 50% off. What is the total profit?", "ground_truth": 1300.0, "problem_type": "multi_step", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "multi_005", "question": "A recipe calls for 2 cups of flour for 12 cookies. You want to make 48 cookies. Each bag of flour has 5 cups. How many bags do you need?", "ground_truth": 2.0, "problem_type": "multi_step", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "multi_006", "question": "A train leaves at 8:00 AM going 60 mph. A car leaves at 9:00 AM going 80 mph from the same point in the same direction. At what time does the car catch the train?", "ground_truth": 11.0, "problem_type": "multi_step", "expected_error_type": "structural", "difficulty": "hard"},
    {"id": "multi_007", "question": "A bank offers 3% annual interest compounded yearly. If you deposit $5000, how much will you have after 4 years? Round to the nearest dollar.", "ground_truth": 5627.0, "problem_type": "multi_step", "expected_error_type": "algorithmic", "difficulty": "hard"},
    {"id": "multi_008", "question": "A teacher has 5 boxes of pencils. Each box has 24 pencils. She gives 8 pencils to each of 12 students. How many pencils are left?", "ground_truth": 24.0, "problem_type": "multi_step", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "multi_009", "question": "A car rental costs $40 per day plus $0.25 per mile. You rent for 3 days and drive 280 miles. What is the total cost?", "ground_truth": 190.0, "problem_type": "multi_step", "expected_error_type": "template", "difficulty": "medium"},
    {"id": "multi_010", "question": "A population of bacteria doubles every hour. Starting with 100 bacteria, how many will there be after 6 hours?", "ground_truth": 6400.0, "problem_type": "multi_step", "expected_error_type": "algorithmic", "difficulty": "medium"},
]


def load_builtin_problems() -> list[Problem]:
    """Load the built-in 50-problem benchmark set."""
    return [Problem.from_dict(d) for d in BUILTIN_PROBLEMS]


def load_problems_from_json(path: str | Path) -> list[Problem]:
    """Load problems from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Problem.from_dict(d) for d in data]


def save_problems_json(problems: list[Problem], path: str | Path) -> None:
    """Save problems to a JSON file."""
    Path(path).write_text(
        json.dumps([p.to_dict() for p in problems], indent=2),
        encoding="utf-8",
    )


def stratified_split(
    problems: list[Problem],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Problem], list[Problem], list[Problem]]:
    """Split problems into train/val/test, stratified by problem_type.

    Ensures each split has proportional representation of each type.
    """
    import random

    rng = random.Random(seed)

    # Group by type
    by_type: dict[str, list[Problem]] = {}
    for p in problems:
        by_type.setdefault(p.problem_type, []).append(p)

    train, val, test = [], [], []
    for ptype, group in by_type.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    return train, val, test


def problems_by_type(problems: list[Problem]) -> dict[str, list[Problem]]:
    """Group problems by type."""
    result: dict[str, list[Problem]] = {}
    for p in problems:
        result.setdefault(p.problem_type, []).append(p)
    return result
