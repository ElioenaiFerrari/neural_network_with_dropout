import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path
from models.neural_model import NeuralModel

path = Path(__file__).parent.absolute()


x = pd.read_csv(f'{path}/inputs.csv')
y = pd.read_csv(f'{path}/outputs.csv')

model = NeuralModel()
classifier = KerasClassifier(
    build_fn=model.create_network,
    epochs=100,
    batch_size=10,
)

results = cross_val_score(
    estimator=classifier,
    X=x,
    y=y,
    cv=10,
    scoring='accuracy'
)

mean = results.mean()
deviation = results.std()


print(deviation)
print(mean)
