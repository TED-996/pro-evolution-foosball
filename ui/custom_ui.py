from sim import simulation
import pymunk
import pymunk.pygame_util
import pygame

_scale = 500


def r_int(value):
    return int(round(value))


def _draw(ball, rods, rod_owners, goals, sides, surface: pygame.Surface):
    ball_shape, = ball.shapes
    assert isinstance(ball_shape, pymunk.Circle)
    _draw_circle(ball_shape, (50, 50, 255), surface, _scale)

    for body in sides:
        for shape in body.shapes:
            assert isinstance(shape, pymunk.Segment)
            _draw_segment(shape, (255, 255, 255), surface, _scale)
    for body, color in zip(goals, ((255, 50, 50), (50, 255, 50))):
        for shape in body.shapes:
            assert isinstance(shape, pymunk.Segment)
            _draw_segment(shape, color, surface, _scale)

    for body, owner in zip(rods, rod_owners):
        color = ((255, 50, 50), (50, 255, 50))[owner]
        for shape in body.shapes:
            assert isinstance(shape, pymunk.Poly)
            _draw_poly(shape, color, surface, _scale)


def _get_table_surface(sim: simulation.Simulation):
    x = r_int(sim.table_info.length * _scale)
    y = r_int(_scale)

    return pygame.Surface((x, y))


def _draw_segment(segment: pymunk.Segment, color, surface, scale):
    orig_a = segment.body.local_to_world(segment.a)
    orig_b = segment.body.local_to_world(segment.b)
    a = (r_int(orig_a[0] * scale), r_int(orig_a[1] * scale))
    b = (r_int(orig_b[0] * scale), r_int(orig_b[1] * scale))
    r = r_int(segment.radius * scale) or 1

    pygame.draw.line(surface, color, a, b, r)


def _draw_circle(circle: pymunk.Circle, color, surface, scale):
    orig_c = circle.body.local_to_world(circle.offset)
    c = (r_int(orig_c[0] * scale), r_int(orig_c[1] * scale))
    r = r_int(circle.radius * scale) or 1

    pygame.draw.circle(surface, color, c, r)


def _draw_poly(poly: pymunk.Poly, color, surface, scale):
    def get_point(vertex):
        x, y = poly.body.local_to_world(vertex)
        return r_int(x * scale), r_int(y * scale)

    points = map(get_point, poly.get_vertices())
    pygame.draw.polygon(surface, color, list(points))


def run(sim: simulation.Simulation, inputs_function):
    pygame.init()

    screen: pygame.Surface = pygame.display.set_mode((1024, 768))
    table: pygame.Surface = _get_table_surface(sim)
    done = False
    clock = pygame.time.Clock()

    table_pos = ((screen.get_width() - table.get_width()) / 2,
                 (screen.get_height() - table.get_height()) / 2)

    rod_owners = sim.get_rod_owners()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        tick_s = clock.tick(60) / 1000.0
        for side, input in inputs_function(tick_s):
            sim.apply_inputs(side, input)
        sim.tick(tick_s)

        screen.fill((0, 0, 0))
        table.fill((0, 0, 0))

        _draw(sim.ball_body, sim.rod_bodies, rod_owners, sim.goal_bodies, sim.side_bodies, table)
        screen.blit(table, table_pos)

        pygame.display.flip()
