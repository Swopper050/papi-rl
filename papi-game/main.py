import arcade
import random
import math
import time

#Set constraints for width & Height
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

#Settings
SPRITE_SCALING_PLAYER = 0.3
movementSpeed = 7
Speed = 10
iniJumpSpeed = 15.5
jumpSpeed = 9
gravity = -1

#Papi stats
startPosPapiX = 600
startPosPapiY = 140
lastJumpHeight = 235
#Onion stats
startPosMonsterX = 500
startPosMonsterY = 185
monsterSpeed = 4
SPRITE_SCALING_MONSTER = 0.5

class Monster(arcade.Sprite):
    monsterType = 0
    monsterPoints = 0
    def update(self):
        #Onion
        #side = 1? monster from left to right
        #side = 0? monster from right to left
        if self.side == 1:
            self.center_x += monsterSpeed
        elif self.side == 0:
            self.center_x -= monsterSpeed

        #Points the monster is worth
        if self.monsterType == 1:
            self.monsterPoints = 100
        elif self.monsterType == 2:
            self.monsterPoints = 130
        elif self.monsterType == 3:
            self.monsterPoints = 170
        elif self.monsterType == 4:
            self.monsterPoints = 200

class Player(arcade.Sprite):


    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y
        self.game_over = False

        #Dont go out of the screen
        if self.left < 0:
            self.left = 0
        elif self.right > SCREEN_WIDTH:
            self.right = SCREEN_WIDTH
        if self.bottom < startPosPapiY:
            self.bottom = startPosPapiY

        #Gravity simulator
        self.change_y += gravity
        if self.top == startPosPapiY:
            self.change_y = 0

#---------------------------------Game Class-----------------------------#
class myGame(arcade.Window):

    def __init__(self, width, height):
        super().__init__(width, height)
        #Set the background color to white
        arcade.set_background_color(arcade.color.BLUEBERRY)


    def setup(self):
        self.score = 0
        self.combo = 1
        self.gameSpeed = 100
        self.time = 0
        self.actualTime = 0

        #Create the Spritelists
        self.player_list = arcade.SpriteList()
        self.monster_list = arcade.SpriteList()
        #Setup the player
        self.player_sprite = Player("images/papiRound.PNG", SPRITE_SCALING_PLAYER)

        #Start position of Papi
        self.player_sprite.center_x = startPosPapiX
        self.player_sprite.center_y = startPosPapiY
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        arcade.start_render()

        self.player_list.draw()
        self.monster_list.draw()

        arcade.draw_rectangle_filled(500, 75, 1000 , 150, arcade.color.GREEN)

        #Show the points
        output = f"Score: {self.score}"
        arcade.draw_text(output, 50, 550, arcade.color.WHITE, 20)

    def update(self, delta_time):
        #50 frames = 1s

        self.time += 1
        self.actualTime += 0.02

        if self.time % 750 == 0:
            self.gameSpeed -= 8
            print("Speeding up")
        gameOver = False
        self.player_list.update()
        self.monster_list.update()


        #Randomly spawn enemies
        if random.randrange(self.gameSpeed) == 0:
            randint = random.randint(0, 1)
            monsterKind = 1
            if self.gameSpeed < 86: monsterKind = random.randint(1, 4)
            elif self.gameSpeed < 94: monsterKind = random.randint(1, 3)
            elif self.gameSpeed < 102: monsterKind = random.randint(1, 2)
            elif self.gameSpeed <= 110: monsterKind = 1
            #what monster do we spawn?
            if monsterKind == 1:
                newMonster = Monster("images/droltrans-2.PNG", SPRITE_SCALING_MONSTER * 0.3375)
                newMonster.center_y = startPosMonsterY - 5
            if monsterKind == 2:
                newMonster = Monster("images/watermelontrans.PNG", SPRITE_SCALING_MONSTER * 0.1)
                newMonster.center_y = startPosMonsterY + 15
            if monsterKind == 3:
                newMonster = Monster("images/heinekentrans.PNG", SPRITE_SCALING_MONSTER * 0.40)
                newMonster.center_y = startPosMonsterY + 115
            if monsterKind == 4:
                newMonster = Monster("images/tomatotrans.PNG", SPRITE_SCALING_MONSTER * 0.4)
                newMonster.center_y= startPosMonsterY + 235
            newMonster.monsterType = monsterKind
            #Left to right || right to left
            if randint == 1:
                newMonster.side = 1
                newMonster.center_x = startPosMonsterX - 500
            else:
                newMonster.side = 0
                newMonster.center_x = startPosMonsterX + 500
            self.monster_list.append(newMonster)
        #Check for collision (score)
        monsterHit_list = self.monster_list
        hit = 0

        for monster in monsterHit_list:
            var = 0
            if monster.monsterType == 3:
                var = 50


            deltaY = self.player_sprite.position[1] - monster.position[1]
            deltaX = monster.position[0] - self.player_sprite.position[0]



            #Check for X Value of papi between monster radius
            if abs(deltaX) < 60:        #Are we standing next to a monster?
                if deltaY < 60 and deltaY > 10: #Are we above the monster?
                    if self.player_sprite.change_y > 0:
                        gameOver = True
                        print("monsterX")
                        print(monster.position[0])
                        print("MonsterY")
                        print(monster.position[1])
                        print("papiX")
                        print(self.player_sprite.position[0])
                        print("papiY")
                        print(self.player_sprite.position[1])
                        print("Over here")
                        break
                    hit = 1
                    self.score += self.combo * monster.monsterPoints
                    self.combo += 1
                    monster.remove_from_sprite_lists()

            #Game Over
            if abs(deltaX) + var < 50 or abs(deltaX) - var < 50:
                if abs(deltaY) < 15:
                    print("monsterX")
                    print(monster.position[0])
                    print("MonsterY")
                    print(monster.position[1])
                    print("papiX")
                    print(self.player_sprite.position[0])
                    print("papiY")
                    print(self.player_sprite.position[1])
                    gameOver = True
        global lastJumpHeight
        if hit == 1:
            #Jump if you are on top of a monster
            lastJumpHeight = self.player_sprite.position[1] - 70
            self.player_sprite.change_y = iniJumpSpeed
        #Reached the ground? Combo resets
        if self.player_sprite.position[1] <=200:
            self.combo = 1
            lastJumpHeight = 135

        if gameOver:
            print("{} {}".format("Final score = ", self.score))
            exit()
            close_window()

    #If a button is pressed
    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.player_sprite.change_x = -movementSpeed
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_x = movementSpeed
        elif key == arcade.key.UP and self.player_sprite.center_y < startPosPapiY + 60:
            self.player_sprite.change_y = iniJumpSpeed

    #If a button is released
    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player_sprite.change_x = 0
#----------------------------------Main----------------------------------#

def main():
    game = myGame(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()
