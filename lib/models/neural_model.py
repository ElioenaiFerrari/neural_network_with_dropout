from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout


class NeuralModel:
    model = Sequential()

    def create_network(
        self,
        optimizer,
        loss,
        kernel_initializer,
        activation,
        neurons
    ):
        self.model.add(
            Dense(
                input_dim=30,
                units=neurons,
                kernel_initializer=kernel_initializer,
                activation=activation,
            )
        )

        self.model.add(Dropout(0.2))

        self.model.add(
            Dense(
                units=neurons,
                kernel_initializer=kernel_initializer,
                activation=activation,
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
            optimizer=optimizer,
            loss=loss,
            metrics=['binary_accuracy']
        )

        return self.model
