from gym.envs.registration import register

register(
    id='ActiveVision-v0',
    entry_point='gym_activevision.envs:ActiveVisionEnv',
)
#register(
#    id='foo-extrahard-v0',
#    entry_point='gym_foo.envs:FooExtraHardEnv',
#)
