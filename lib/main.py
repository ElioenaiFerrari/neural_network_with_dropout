import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from pathlib import Path
from models.neural_model import NeuralModel

path = Path(__file__).parent.absolute()


x = pd.read_csv(f'{path}/inputs.csv')
y = pd.read_csv(f'{path}/outputs.csv')

model = NeuralModel()
classifier = KerasClassifier(build_fn=model.create_network)

params = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'optimizer': ['adam', 'sgd'],
    'loss': ['binary_crossentropy', 'hinge'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [16, 8]
}

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=params,
    scoring='accuracy',
    cv=5
)

grid_search = grid_search.fit(x, y)
best_params = grid_search.best_params_
best_precision = grid_search.best_score_

print(best_params)
print(best_precision)

# results = cross_val_score(
#     estimator=classifier,
#     X=x,
#     y=y,
#     cv=10,
#     scoring='accuracy'
# )

# mean = results.mean()
# deviation = results.std()


# print(deviation)
# print(mean)
