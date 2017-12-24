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


def _draw(space:pymunk.Space, surface:pygame.Surface):
    options = pymunk.pygame_util.DrawOptions(surface)
    pymunk.pygame_util.positive_y_is_up = False
    space.debug_draw(options)


def run(sim:simulation.Simulation, inputs_function):
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        for side, input in inputs_function():
            sim.apply_inputs(side, input)
        sim.tick(clock.tick(60) / 1000.0)

        _draw(sim.space, screen)
        pygame.display.flip()
