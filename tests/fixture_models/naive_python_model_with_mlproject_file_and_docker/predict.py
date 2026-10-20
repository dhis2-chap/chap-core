import sys
print("predicting")

model_name = sys.argv[1]
model = open(model_name, "r").read()
print("Modek is", model)
