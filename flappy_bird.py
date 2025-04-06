import pygame
import time
import os 
import random


#When you name variable with capslock, it turn to a constant
WIN_WIDTH = 600
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird3.png")))]


PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img", "bg.png")))


#Bird Object
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5
    
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0 #rotation count
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0] # load first img when initiate bird
        
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        
    def move(self):
        self.tick_count += 1
        
        #gravity simulate(displacement) 
        # a = 1, t = tick_count
        d = self.vel*self.tick_count + 1.5*self.tick_count ** 2
        
        if d >= 16:
            d = 16
            
        self.y = self.y + d
        
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION: # prevent the bird to spin around
                self.tilt = self.MAX_ROTATION
            
            else:
                if self.tilt > -90:
                    self.tilt -= self.ROT_VEL
                    
    def draw(self, win):
        self.img_count += 1 #How much loop done 
        
        #wing flap movement
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
            
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
            
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
            
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]

        elif self.img_count < self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        if self.tilt <= -80:
            self.img = self.img[1]
            self.img_count = self.ANIMATION_TIME * 2
            

        #rotate bird img
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)
        
        
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    
#Pipe Object
















#----RUNNING SECTION----
def draw_window(win, bird):
    win.blit(BG_IMG,(0,0))
    bird.draw(win)
    pygame.display.update()


def main():
    bird = Bird(200,200)
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:        
                run = False
        draw_window(win,bird)
        bird.move()
    pygame.quit()
    quit()
    

pygame.init()
if __name__ == "__main__":
    main()