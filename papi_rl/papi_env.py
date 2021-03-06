import random
from enum import IntEnum

import gym
import numpy as np
import pygame
from pygame.locals import (
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_RIGHT,
    K_UP,
    KEYDOWN,
    QUIT,
    RLEACCEL,
)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
INITIAL_JUMP_VELOCITY = 20

N_FRAMES_PER_OBS = 4


class PapiAction(IntEnum):
    nothing = 0
    left = 1
    right = 2
    jump = 3


class PapiEnv(gym.Env):
    def __init__(self, normal_speed=False):
        """
        :param normal_speed: bool, whether to render the environment at 40 fps
        """
        self.normal_speed = normal_speed

        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(N_FRAMES_PER_OBS, SCREEN_HEIGHT, SCREEN_WIDTH)
        )
        self.action_space = gym.spaces.Discrete(4)

        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.score_font = pygame.font.SysFont("monospace", 16)

        self.reset()

    def reset(self):
        """
        Resets the game and returns the initial observation.

        :returns: np.ndarray with the initial observation
        """
        self.player = PapiPlayer()
        self.player.set_initial_position()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)
        self.enemies = pygame.sprite.Group()

        self.total_score = 0
        self.timestep = 0
        self.stacked_obs = np.zeros((N_FRAMES_PER_OBS, SCREEN_HEIGHT, SCREEN_WIDTH))
        return self.get_observation()

    def step(self, action):
        """
        Executes the action in the environment. This is a left, right or up key press.

        :param action: int, the action to execute
        :returns: the new observation, the reward, whether the episode is done and an
                  info dictionary which will be empty
        """
        self.timestep += 1
        if self.normal_speed:
            self.clock.tick(40)

        monsters_to_add = []
        if self.timestep > 100 and random.uniform(0, 1) < 0.005:
            monsters_to_add.append(MonsterType.onion.value)
        if self.timestep > 1250 and random.uniform(0, 1) < 0.005:
            monsters_to_add.append(MonsterType.watermelon.value)
        if self.timestep > 2500 and random.uniform(0, 1) < 0.005:
            monsters_to_add.append(MonsterType.carrot.value)
        if self.timestep > 5000 and random.uniform(0, 1) < 0.005:
            monsters_to_add.append(MonsterType.tomato.value)

        for monster_type in monsters_to_add:
            monster = PapiMonster(monster_type)
            self.enemies.add(monster)
            self.all_sprites.add(monster)

        self.player.update(action)
        self.enemies.update()

        done = False
        reward = 0
        collisions = pygame.sprite.spritecollide(self.player, self.enemies, False)
        if len(collisions) > 0:
            for monster_hitted in collisions:
                dy = self.player.rect.bottom - monster_hitted.rect.top
                if self.player.moving_down and dy < 30:
                    self.player.start_jump()

                    reward += self.player.combo * monster_hitted.points
                    self.player.combo += 1
                    monster_hitted.kill()
                elif dy > 40:
                    self.player.kill()
                    reward -= 10000
                    done = True

        self.total_score += reward

        self.update_screen()
        return self.get_observation(), reward / 100, done, {}

    def update_screen(self):
        """ Redraws the game screen with all entities and score. """
        self.screen.fill((0, 0, 0))

        for entity in self.all_sprites:
            self.screen.blit(entity.surf, entity.rect)

        scoretext = self.score_font.render(
            "Score: {}".format(self.total_score), 1, (255, 255, 255)
        )
        self.screen.blit(scoretext, (5, 10))

        combotext = self.score_font.render(
            "Combo: {}".format(self.player.combo - 1), 1, (255, 255, 255)
        )
        self.screen.blit(combotext, (5, 30))

    def render(self):
        """
        Renders the game. Quite expensive, so should not be used during training.
        """
        pygame.display.update()

    def get_observation(self):
        """
        :returns: 3d np.ndarray with the current obervation
        """
        current_obs = pygame.surfarray.array2d(self.screen).reshape(
            (1, SCREEN_HEIGHT, SCREEN_WIDTH)
        )
        self.stacked_obs = np.concatenate(
            (self.stacked_obs[1:, :, :], current_obs), axis=0
        )
        return self.stacked_obs


class PapiPlayer(pygame.sprite.Sprite):
    def __init__(self):
        super(PapiPlayer, self).__init__()
        player_im = pygame.image.load("papi_rl/img/papi.png")
        self.surf = pygame.transform.scale(player_im, (100, 100)).convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect()
        self.combo = 1

        self.reset_jump()
        self.set_initial_position()

    def set_initial_position(self):
        self.rect.x = SCREEN_WIDTH / 2
        self.rect.y = SCREEN_HEIGHT

    def update(self, action):
        """
        Updates the position of the player given the current action.

        :param action: int, current action
        """
        if action == PapiAction.left.value:
            self.rect.move_ip(-15, 0)
        elif action == PapiAction.right.value:
            self.rect.move_ip(15, 0)

        if not self.currently_jumping:
            if action == PapiAction.jump.value:
                self.start_jump()

        if self.currently_jumping:
            self.rect.move_ip(0, -self.jump_velocity)

            self.jump_velocity = max(self.jump_velocity - 1, -INITIAL_JUMP_VELOCITY)

            if self.jump_velocity <= 0:
                self.moving_down = True

            if self.rect.bottom >= SCREEN_HEIGHT:
                self.reset_jump()

        # Make sure the player stays within the screen
        self.rect.left = max(self.rect.left, 0)
        self.rect.right = min(self.rect.right, SCREEN_WIDTH)
        self.rect.bottom = min(self.rect.bottom, SCREEN_HEIGHT)
        self.rect.top = max(self.rect.top, 0)

    def reset_jump(self):
        self.combo = 1
        self.currently_jumping = False
        self.moving_down = False
        self.jump_velocity = INITIAL_JUMP_VELOCITY

    def start_jump(self):
        self.currently_jumping = True
        self.moving_down = False
        self.jump_velocity = INITIAL_JUMP_VELOCITY


class MonsterType(IntEnum):
    onion = 0
    watermelon = 1
    carrot = 2
    tomato = 3


class PapiMonster(pygame.sprite.Sprite):
    def __init__(self, monster_type):
        super(PapiMonster, self).__init__()
        if monster_type == MonsterType.onion.value:
            monster_im = pygame.image.load("papi_rl/img/onion.png")
            monster_im = pygame.transform.scale(monster_im, (100, 100))
            monster_height = SCREEN_HEIGHT - 100
            self.speed = 5 + random.uniform(-2, 2)
            self.points = 100
        elif monster_type == MonsterType.watermelon.value:
            monster_im = pygame.image.load("papi_rl/img/watermelon.png")
            monster_im = pygame.transform.scale(monster_im, (100, 100))
            monster_height = SCREEN_HEIGHT - 100
            self.speed = 7 + random.uniform(-3, 3)
            self.points = 200
        elif monster_type == MonsterType.carrot.value:
            monster_im = pygame.image.load("papi_rl/img/carrot.png")
            monster_im = pygame.transform.scale(monster_im, (150, 40))
            monster_height = SCREEN_HEIGHT - 270
            self.speed = 5 + random.uniform(-3, 3)
            self.points = 300
        elif monster_type == MonsterType.tomato.value:
            monster_im = pygame.image.load("papi_rl/img/tomato.png")
            monster_im = pygame.transform.scale(monster_im, (100, 100))
            monster_height = SCREEN_HEIGHT - 440
            self.speed = 5 + random.uniform(-3, 3)
            self.points = 500

        self.direction = -1 if random.uniform(0, 1) < 0.5 else 1
        if random.uniform(0, 1) < 0.5:
            self.direction = 1
            start_x = -200
        else:
            monster_im = pygame.transform.flip(monster_im, True, False)
            self.direction = -1
            start_x = SCREEN_WIDTH + 200

        self.surf = monster_im.convert()
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.surf.get_rect()
        self.rect.x = start_x
        self.rect.y = monster_height

        self.speed += pygame.time.get_ticks() / 5e5

    def update(self):
        """
        Updates the position of the player given the current action.

        :param action: int, current action
        """

        self.rect.move_ip(self.direction * self.speed, 0)
        if self.direction == 1 and self.rect.left > SCREEN_WIDTH:
            self.kill()
        elif self.direction == -1 and self.rect.right < 0:
            self.kill()
