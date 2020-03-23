from utils import FP, get_tf_env
import pbt_runner
import population

POPULATION_SIZE = 10
NUM_TO_EVOLVE = 3
NUM_EPOCHS = 1e4
NUM_ENV_STEPS_PER_EPOCH = 1e6


def main():
    tf_env = get_tf_env()
    members = population.create_members(POPULATION_SIZE)
    hyperparams = FP.PPO_HYPERPARAMS
    pop = population.Population(members, hyperparams)
    pbt = pbt_runner.PBTRunner(population=pop, env=tf_env, num_to_evolve=NUM_TO_EVOLVE)
    pbt.run_pbt(root_dir='./tmp/', num_epochs=NUM_EPOCHS, num_env_steps_per_epoch=NUM_ENV_STEPS_PER_EPOCH)

if __name__=='__main__':
    main()