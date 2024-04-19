from annotated_types import Interval
from typing import Annotated

Probability = Annotated[float, Interval(ge=0, le=1)]
Positive = Annotated[float, Interval(gt=0)]
Rate = Positive


class GlobalParams:
    infected_decay: Probability
    observation_rate: Rate
    state_sigma: Positive

class StateParams:
    infected_proportion: Probability


def observation_distribution(state: StateParams, global_params: GlobalParams, predictors: dict[str, Any]) -> IsDistribution:
    return PoissonSkipNaN(state.infected * global_params.observation_rate * predictors['population'])

def state_distribution(previous_raw_state: StateSpace[StateParams], params: GlobalParams, predictors: Any=None) -> Distribution[StateParams]:
    mu = previous_raw_state.infected_proportion * params.infected_decay
    return StateDistribution(Normal(mu, global_params.state_sigma))



default_transforms = {Probability: 'expit',
                      Rate: 'exp',
                      Positive: 'exp'
                      float: 'identity'}

default_priors = {Probability: 'Logistic(0, 1)',
                  Rate: 'Normal(0, 10.)',
                  Positive: 'Normal(0, 10.)'}




def from_dict

print(default_transforms[Probability])