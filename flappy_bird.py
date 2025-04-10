import neat.nn.feed_forward
import neat.population
import pygame
import time
import os 
import random
import neat

pygame.font.init()

#When you name variable with capslock, it turn to a constant
WIN_WIDTH = 600
WIN_HEIGHT = 800

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("img","bird3.png")))]


PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img", "bg.png")))
SCORE_DONT = pygame.font.SysFont("comicsans",50)


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
class Pipe:
    GAP = 200
    VEL = 5
    
    def __init__(self,x):
        self.x = x
        self.height = 0
        self.gap = 100
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        
        self.passed = False
        self.set_height()
        
    #set height for pipe     
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
        
    #pipe will move reverse to the velocity     
    def move(self):
        self.x -= self.VEL
    
    #draw pipe
    def draw(self,win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
        
    #Use pixel coordinate coincidence to simulate collision, check pixes up against each other
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        
        top_offsets = (self.x - bird.x , self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offsets)
        
        if t_point or b_point:
            return True
        
        return False
        
        
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG
    
    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
            
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
        
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
        
        
#----RUNNING SECTION----
def draw_window(win, bird,pipes, base, score):
    win.blit(BG_IMG,(0,0))
    for pipe in pipes:
        pipe.draw(win)
        
    text = SCORE_DONT.render("Score: " + str(score), 1, (255,255,255))    
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(),10))
        
    base.draw(win)
    bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    nets = []
    ge =[] # generation count 
    birds = []
    
    for g in genomes:
        net = neat.nn.FeedForwardNetwork(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)
    
    bird = Bird(230,350)
    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()
    
    score = 0
    
    run = True
    
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:        
                run = False
                
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
                
       
        # bird.move()
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.remove(bird)
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                     
                
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
                    
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            pipe.move()
            
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(700))
            
        for r in rem:
            pipes.remove(r)
           
        for bird in birds: 
            if bird.y + bird.img.get_height() >= 730: 
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
            
            
        base.move()
        draw_window(win,bird,pipes,base)
    pygame.quit()
    quit()
    

pygame.init()
def run(config_path):
    config = neat.config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(config)#initiate population
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main,50)


if __name__ == "__main__":
    main()
    local_path = os.path(__file__)
    config_path= os.path.join(local_path, "config_feedfoward.txt")