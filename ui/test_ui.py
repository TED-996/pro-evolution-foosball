from sim import simulation
import pymunk
import pymunk.pygame_util
import pygame


"""
IMPORTANT:
There is still a problem: 
Our space uses coordinates mostly in (0, 1) and i suspect that 
pygame takes these values as PIXELS.
So our entire space is a pixel.
Might want to fix that.
(scale the space or something)
"""


def hack_the_pymunk():
    from_pygame_save = pymunk.pygame_util.from_pygame
    to_pygame_save = pymunk.pygame_util.to_pygame

    scale = 500

    def from_pygame(p, surface):
        x, y = p
        #return from_pygame_save((x / 1000, y / 1000), surface)
        return x / scale, y / scale

    def to_pygame(p, surface):
        x, y = p
        #return to_pygame_save((x * 1000, y * 1000), surface)
        return int(x * scale), int(y * scale)


    pymunk.pygame_util.from_pygame = from_pygame
    pymunk.pygame_util.to_pygame = to_pygame


hack_the_pymunk()


def _draw(space: pymunk.Space, surface: pygame.Surface):
    options = pymunk.pygame_util.DrawOptions(surface)
    pymunk.pygame_util.positive_y_is_up = False
    space.debug_draw(options)


def run(sim: simulation.Simulation, inputs_function):
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        tick_s = clock.tick(60) / 1000.0
        for side, input in inputs_function(tick_s):
            sim.apply_inputs(side, input)
        sim.tick(tick_s)

        screen.fill((0, 0, 0))
        _draw(sim.space, screen)
        pygame.display.flip()
