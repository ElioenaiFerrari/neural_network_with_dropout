from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


class NeuralModel:
    model = Sequential()

    optimizer = optimizers.Adam(
        lr=0.001,
        decay=0.0001,
        clipvalue=0.5
    )

    def create_network(self):
        self.model.add(
            Dense(
                input_dim=30,
                units=16,
                kernel_initializer='random_uniform',
                activation='relu',
            )
        )

        self.model.add(Dropout(0.2))

        self.model.add(
            Dense(
                units=16,
                kernel_initializer='random_uniform',
                activation='relu',
            )
        )

        self.model.add(Dropout(0.2))

        self.model.add(
            Dense(
                units=1,
                activation='sigmoid'
            )
        )

        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )

        return self.model
