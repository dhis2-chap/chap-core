import sys
future_data = sys.argv[1]
model_file = sys.argv[2]
out_file = sys.argv[3]

print("Predict")
sys.exit()

with open(future_data) as f:
    with open(out_file, "w") as out:
        out.write(f.read())

