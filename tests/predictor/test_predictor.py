import pandas as pd
from chap_core.predictor.poisson import Poisson


def test_predictor():
    poisson = Poisson()
    sample_x = pd.DataFrame(
        {
            "Disease1": [1, 2, 3, 4, 5],
            "Disease2": [1, 2, 3, 4, 5],
            "Rain": [1, 2, 3, 4, 5],
            "Temperature": [1, 2, 3, 4, 5],
        }
    )
    sample_y = pd.DataFrame(
        {
            "Disease": [1, 2, 3, 4, 5],
        }
    )

    poisson.train(sample_x, sample_y)
    prediction = poisson.predict(sample_x)
