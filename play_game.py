import argparse

import pygame
from pygame.locals import K_ESCAPE, K_LEFT, K_RIGHT, K_UP, KEYDOWN, QUIT

from papi_rl import PapiAction, PapiEnv


def get_action(prev_action, is_jumping):
    pressed_keys = pygame.key.get_pressed()

    # If we are already jumping we do not jump again
    if not is_jumping and pressed_keys[K_UP]:
        return PapiAction.jump.value

    # Both left and right yields no action
    if pressed_keys[K_LEFT] and pressed_keys[K_RIGHT]:
        return PapiAction.nothing.value

    if pressed_keys[K_LEFT]:
        return PapiAction.left.value

    if pressed_keys[K_RIGHT]:
        return PapiAction.right.value
    return PapiAction.nothing.value


def main(args):
    env = PapiEnv(normal_speed=True)

    obs = env.reset()
    env.render()

    done = False
    prev_action = PapiAction.nothing.value
    while not done:

        stop_game = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_game = True

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    stop_game = True

        action = get_action(prev_action, env.player.currently_jumping)
        obs, reward, env_done, _ = env.step(action)
        env.render()

        done = env_done or stop_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
