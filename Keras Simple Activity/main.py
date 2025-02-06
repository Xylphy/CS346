from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt("Keras Simple Activity/pima-indians-diabetes.data.csv", delimiter=",")

X = dataset[:, :8]
Y = dataset[:, 8]

model = Sequential()

model.add(Dense(12, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, Y) # returns loss and accuracy
print("Accuracy: %.2f" % (accuracy * 100))

predictions = model.predict(X)

print("Predictions: ", predictions[:5])

rounded = [round(x[0]) for x in predictions]

for i in range(5):
    print("%s => %d (expected %d)" % (X[i].tolist(), rounded[i], Y[i]))