import sys

train_data = sys.argv[1]
model_name = sys.argv[2]

print(sys.argv)

with open(model_name, "w") as f:
    f.write("test")



