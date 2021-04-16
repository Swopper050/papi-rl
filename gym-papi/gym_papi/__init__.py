from gym.envs.registration import register

register(
    id="papi-v0",
    entry_point="gym_papi.envs:PapiEnv",
)
