"""Reproduce the paper's TFLite accuracy from the shipped model and test CSV.

Usage:
    pip install -r requirements.txt
    python verify_model.py

Runs model/pose_classifier.tflite over the 571 held-out samples in
data/test2_data.csv and reports accuracy. Expected: 562/571 = 0.9842.
"""
import csv
import os

import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "pose_classifier.tflite")
TEST_CSV = os.path.join(HERE, "data", "test2_data.csv")
LABELS = open(os.path.join(HERE, "model", "pose_labels.txt")).read().splitlines()

interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"model : input {inp['shape'].tolist()}, output {out['shape'].tolist()}")

with open(TEST_CSV, encoding="utf-8") as f:
    rows = list(csv.reader(f))[1:]
print(f"data  : {len(rows)} test samples, {len(LABELS)} classes")

per_class = {i: [0, 0] for i in range(len(LABELS))}  # correct, total
correct = 0
for r in rows:
    vec = np.array([[float(v) for v in r[1:52]]], dtype=np.float32)
    interpreter.set_tensor(inp["index"], vec)
    interpreter.invoke()
    pred = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
    truth = int(r[52])
    per_class[truth][1] += 1
    if pred == truth:
        per_class[truth][0] += 1
        correct += 1

print("\nper-class accuracy:")
for i, name in enumerate(LABELS):
    c, n = per_class[i]
    print(f"  {name:<40s} {c:>3d}/{n:<3d} ({c / n:.2%})")

print(f"\noverall: {correct}/{len(rows)} = {correct / len(rows):.4f}")
