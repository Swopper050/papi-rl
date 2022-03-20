import argparse

from papi_rl import train_agent


def main(args):
    train_agent(n_workers=1, n_epochs=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
