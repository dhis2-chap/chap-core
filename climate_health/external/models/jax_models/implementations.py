from climate_health.external.models.jax_models.model_spec import PoissonSkipNaN
from climate_health.external.models.jax_models.protoype_annotated_spec import Probability, Rate, Positive
'''
random variables: should behave exactly like a sample from that distribution
'''

class GlobalParams:
    infected_decay: Probability
    observation_rate: Rate
    state_sigma: Positive


class StateParams:
    infected_proportion: Probability


class Inputs:
    population: int


class PopulationPoisson:
    params: GlobalParams
    state: StateParams
    inputs: Inputs

    def _dist(self):
        return PoissonSkipNaN(self.state.infected_proportion * self.params.observation_rate * self.inputs.population)

    def sample(self, shape: tuple[int]) -> int:
        self._dist.sample(shape)

    def log_prob(self, value: int) -> float:
        self._dist.log_prob(value)


class InfectedDistribution:
    params: GlobalParams
    state: StateParams
    inputs: Inputs

    def _dist(self):
        return Normal(self.state.infected_proportion * self.params.infected_decay, self.params.state_sigma)

    def sample(self, shape: tuple[int]) -> int:
        self._dist.sample(shape)

    def log_prob(self, value: int) -> float:
        self._dist.sample(value)

'''

S[t+1] = Normal(S[t]*params.infected_decay, params.state_sigma)
T = transformation(S)
I = PoissonSkipNaN(T*params.observation_rate*population)


T[t+1] = MyState(human=Multinomial(T[t].human@params.transition_matrix), mosquito=Multinomial(T[t]@params.transition_matrix)) 

I = PoissonSkipNaN(T[3]*params.observation_rate*population)
'''



def state_distribution(previous_raw_state: StateSpace[StateParams], params: GlobalParams, predictors: Any = None) -> \
Distribution[StateParams]:
    mu = previous_raw_state.infected_proportion * params.infected_decay
    return StateDistribution(Normal(mu, global_params.state_sigma))


def observation_distribution(state: StateParams, global_params: GlobalParams,
                             predictors: dict[str, Any]) -> IsDistribution:
    return PoissonSkipNaN(state.infected * global_params.observation_rate * predictors['population'])
