from keras import models
from keras import layers
import numpy as np

models.InputLayer
model = models.Sequential()
model.add(layers.Dense(32, input_shape=(10,)))
model.add(layers.Dense(5))
x = model.predict(np.array([[1,1,1,1,1,2,2,2,2,2],[1,1,1,1,1,2,2,2,2,2]]))

print (x)

