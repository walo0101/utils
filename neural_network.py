from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(100, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu')) # train_X has shape (n,1,m) 
model.add(Dropout(0.005))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.005))
model.add(Dense(train_Y.shape[2], activation='sigmoid')) # train_Y has shape (n,1,k)

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3))
history = model.fit(train_X, train_Y, epochs=50, batch_size=100, verbose=2, validation_split=0.2)
