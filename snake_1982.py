from pygame.locals import *
from random import randint
import pygame
import time
import random

class Player:
#640x400 resolution. 80x25 characters
    step = 16
    direction = 99
    updateCountMax = 2
    updateCount = 0
    x = [0]
    y = [0]
    def __init__(self, id, x,y):
       # initial positions, no collision.
       if id == 0: #player 1
           x[0] = 3*self.step
           y[0] = 10*self.step
           x.append(4*self.step)
           y.append(10*self.step)
           x.append(5*self.step)
           y.append(10*self.step)
           self.direction = 0
           self.x = x
           self.y=y
           #for i in range(0,len(self.x)):
           #     print (self.x[i])
           #print ("----")
                          
       else:#player2
           x[0] = 35*self.step
           y[0] = 10*self.step
           x.append(34*self.step)
           y.append(10*self.step)
           x.append(33*self.step)
           y.append(10*self.step)
           self.direction = 1
           self.x=x
           self.y=y
           #for i in range(0,len(self.x)):
           #    print (self.x[i])
                
    def update(self):
        length = len(self.x)
        # update position of head of snake
        if self.direction == 0:
            self.x.append(self.x[length-1] + self.step)
            self.y.append(self.y[length-1])
        if self.direction == 1:
            self.x.append(self.x[length-1] - self.step)
            self.y.append(self.y[length-1])
        if self.direction == 2:
            self.x.append(self.x[length-1])
            self.y.append(self.y[length-1] - self.step)
        if self.direction == 3:
            self.x.append(self.x[length-1])
            self.y.append(self.y[length-1] + self.step)
        #print(self.x[length-1],",",self.y[length-1],self.direction)
 
    def moveRight(self):
        self.direction = 0
 
    def moveLeft(self):
        self.direction = 1
 
    def moveUp(self):
        self.direction = 2
 
    def moveDown(self):
        self.direction = 3 
 
    def draw(self, surface, image):
        for i in range(0,len(self.x)-1):
            surface.blit(image,(self.x[i],self.y[i])) 
 
class Game:
    def isCollision(self,me,oppo):
        #player1 is 1 longer than player2
        headme = len(me.x)-1
        headoppo = len(oppo.x)-1
        if headme > headoppo:
            head = headoppo
        else:
            head = headme
        #loop through the snake and see if the head position is the same as any block
        for i in range(0,head):
            #do I hit myself
            if me.x[i] == me.x[head] and me.y[i] == me.y[head]:
                return True
            #do I hit my opponent
            if me.x[head] == oppo.x[i] and me.y[head] == oppo.y[i]:
                return True
        #test the extra. This will never happen on the shorter one
        #Impossible to hit myself on the head, so just test hitting the head of the oppo
        if me.x[headme] == oppo.x[headoppo] and me.y[headme]==oppo.x[headoppo]:
           return True
        #see if the head hits a wall
        if me.x[head] == 0 or me.x[head]==624 or me.y[head]==0 or me.y[head]==384:
            #print(me.x[head],",",me.y[head])
            return True
        return False
    
 
class App:
    random.seed()
    windowWidth = 640
    windowHeight = 400
    player = 0
 
    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self.game = Game()
        x1=[0]
        y1=[0]
        x2=[0]
        y2=[0]
        self.player1 = Player(0,x1,y1)
        self.player2 = Player(1,x2,y2)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
 
        pygame.display.set_caption('Snake 1982')
        self._running = True
        image1 = pygame.image.load("redblock.jpg").convert()
        image2 = pygame.image.load("blueblock.jpg").convert()
        image_size = (16,16)
        image1 = pygame.transform.scale(image1,image_size)
        image2 = pygame.transform.scale(image2,image_size)
        self._image_surf1 = image1
        self._image_surf2 = image2
        
    def on_event(self, event):
        if event.type == QUIT:
            self._running = False
 
    def on_loop(self):
        
        #print("player 1")
        self.player1.update()
        #print("player 2")
        self.player2.update()
        if self.game.isCollision(self.player1,self.player2):
            print("You lose!")
            self.on_render()
            self._running = False
            exit()
        if self.game.isCollision(self.player2,self.player1):
            print("You win!")
            self.on_render()
            self._running = False
            exit()
        pass
 
    def on_render(self):
        #print("on render")
        self._display_surf.fill((0,0,0))
        #draw border
        bord_color = (255,255,0)
        pygame.draw.rect(self._display_surf, bord_color, pygame.Rect(0,0 ,640, 400),width=31)
        pygame.display.flip()
 
        self.player1.draw(self._display_surf, self._image_surf1)
        self.player2.draw(self._display_surf, self._image_surf2)
        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()

    def willCollide(self,me,oppo,direction):# will I collide in the next turn in this direction
       head=len(me.x)-1    
       if direction == 0:
          newpos_x = me.x[head]+me.step
          newpos_y = me.y[head]
       elif direction == 1:
          newpos_x = me.x[head]-me.step
          newpos_y = me.y[head]
       elif direction == 2:
          newpos_x = me.x[head]
          newpos_y = me.y[head]-me.step
       else:#direction = 3
          newpos_x = me.x[head]
          newpos_y = me.y[head]+me.step
       #print ("current: ",me.x[head],",",me.y[head],",d=",me.direction)
       #print ("newpos: ",newpos_x,",",newpos_y,",d=",direction)
       for i in range(0,head):
       #will I hit myself
          if newpos_x == me.x[i] and newpos_y == me.y[i]:
             #print("True")
             return True
       #do I hit my opponent
          if newpos_x == oppo.x[i] and newpos_y == oppo.y[i]:
             #print ("True")
             return True
         #see if it will hit a wall
          if newpos_x == 0 or newpos_x==624 or newpos_y==0 or newpos_y==384:
             #print ("True")
             return True
       #print ("False")
       return False  

    def on_execute(self):
        #print("on execute")
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            pygame.event.pump()
            keys = pygame.key.get_pressed() 
 
            if (keys[K_RIGHT]):
                if self.player1.direction !=1:
                   self.player1.moveRight()
 
            if (keys[K_LEFT]):
                if self.player1.direction !=0:
                   self.player1.moveLeft()
 
            if (keys[K_UP]):
                if self.player1.direction !=3:
                   self.player1.moveUp()

            if (keys[K_DOWN]):
                if self.player1.direction !=2:
                   self.player1.moveDown()
 
            if (keys[K_ESCAPE]):
                self._running = False
            #Keep Player 2 from running into a wall if it can avoid it
            if self.willCollide(self.player2,self.player1,self.player2.direction):
               # turn randomly
               if self.player2.direction == 0 or self.player2.direction == 1:#moving horz
                  if random.getrandbits(1):
                      if not self.willCollide(self.player2,self.player1,2): #if it won't collide in this direction
                         self.player2.direction = 2
                      else: #if it will collide go the other way
                         self.player2.direction = 3
                  else:
                      if not self.willCollide(self.player2,self.player1,3): #if it won't collide in this direction
                         self.player2.direction = 3
                      else: #if it will collide go the other way
                         self.player2.direction = 2
               else:#moving vert
                  if random.getrandbits(1):
                     if not self.willCollide(self.player2,self.player1,0):
                        self.player2.direction = 0
                     else: 
                        self.player2.direction = 1
                  else:
                     if not self.willCollide(self.player2,self.player1,1):
                        self.player2.direction = 1
                     else:
                        self.player2.direction = 0

            self.on_loop()            
            self.on_render()

            time.sleep (50.0 / 1000.0);
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
