from . import state
import pymunk
import math
import random


class TableInfo:
    """
    Has:
        A length (the maximum X coordinate, Y is clamped [0, 1]
        For each rod:
            its owner
            x coord (double)
            foosman count (int)
            distance between foosman centers (in [0, 1) )
             => side distance (lateral movement)
        Foosman width, length and height (double in [0, 1) )
        Goal width (in [0, 1) )
        Ball radius
    """

    def __init__(self, length, rods, foosman_size, goal_width, ball_radius):
        self.length = length
        self.rods = []
        for owner, x, foo_count, foo_dist in rods:
            self.rods.append((
                owner,
                x,
                foo_count,
                foo_dist,
                1 - (foo_count - 1) * foo_dist - foosman_size[1]
            ))
        foo_x, foo_y, foo_h = foosman_size
        self.foosman_size = (float(foo_x), float(foo_y), float(foo_h))
        self.goal_width = goal_width
        self.ball_radius = ball_radius

    @staticmethod
    def from_dict(value):
        return TableInfo(
            value["length"],
            [
                (
                    rod["owner"],
                    rod["x"],
                    rod["foo_count"],
                    rod["foo_spacing"]
                )
                for rod in value["rods"]
            ],
            (
                value["foo"]["w"],
                value["foo"]["l"],
                value["foo"]["h"]
            ),
            value["goal_w"],
            value["ball_r"]
        )

    def get_init_state(self):
        return state.GameState(
            (0, 0),
            (complex(0.5, 0.5), complex(0, 0)),
            [((0.5, 0.0), (0.0, 0.0)) for _ in self.rods]
        )

    def get_space(self):
        """
        Returns a pymunk Space with the ball and rods etc.
        and a dict of bodies in the form:
        {
            "ball": ball_body,
            "rods":[
                rod_0_body,
                rod_1_body,
                ...
                rod_n_body
            ],
            "goals": [
                goal_0_body,
                goal_1_body
            ]
        }
        """
        def small_rand(maximum):
            return random.random() * 2 * maximum - maximum

        space = pymunk.Space()
        space.gravity = (0, 0)

        ball_body = pymunk.Body(0, 0)
        ball_shape = pymunk.Circle(ball_body, self.ball_radius)
        ball_shape.density = 1.0
        ball_shape.elasticity = 0.8
        ball_body.position = (self.length / 2 + small_rand(0.05), 0.5 + small_rand(0.05))
        ball_body.velocity = (small_rand(0.05), small_rand(0.05))

        space.add(ball_body, ball_shape)

        rod_bodies = []

        foo_w, foo_h, _ = self.foosman_size

        # noinspection PyShadowingNames
        def create_rect(body, x, y, w, h):
            # shape = pymunk.Poly.create_box(body, (w, h))
            # shape.update(pymunk.Transform(tx=x, ty=y))
            hw = w / 2
            hh = h / 2
            return pymunk.Poly(body, [
                (x - hw, y - hh),
                (x - hw, y + hh),
                (x + hw, y + hh),
                (x + hw, y - hh)
            ])

        for owner, x, foo_count, foo_dist, max_offset in self.rods:
            rod_body = pymunk.Body(0, 0, pymunk.Body.KINEMATIC)
            rod_body.position = (x, 0.5)
            space.add(rod_body)

            rod_bodies.append(rod_body)

            for body_idx in range(foo_count):
                delta_y = foo_dist * ((foo_count - 1) / 2 - body_idx)

                foo_shape = create_rect(rod_body, 0, delta_y, foo_w, foo_h)
                space.add(foo_shape)

        goal0_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        goal0_body.position = (0, 0.5)
        goal0_shape = pymunk.Segment(goal0_body, (0, -self.goal_width / 2), (0, self.goal_width / 2), 0.01)

        goal1_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        goal1_body.position = (self.length, 0.5)
        goal1_shape = pymunk.Segment(goal1_body, (0, -self.goal_width / 2), (0, self.goal_width / 2), 0.01)

        space.add(goal0_body, goal0_shape)
        space.add(goal1_body, goal1_shape)

        side_top_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_top_body.position = (self.length / 2, 0)
        side_top_shape = pymunk.Segment(side_top_body, (-self.length / 2, 0), (self.length / 2, 0), 0.01)
        space.add(side_top_body, side_top_shape)

        side_bottom_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_bottom_body.position = (self.length / 2, 1)
        side_bottom_shape = pymunk.Segment(side_bottom_body, (-self.length / 2, 0), (self.length / 2, 0), 0.01)
        space.add(side_bottom_body, side_bottom_shape)

        goal_clearance = (1 - self.goal_width) / 2
        
        side_lt_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_lt_body.position = (0, goal_clearance / 2)
        side_lt_shape = pymunk.Segment(side_lt_body, (0, -goal_clearance / 2), (0, goal_clearance / 2), 0.01)
        space.add(side_lt_body, side_lt_shape)

        side_lb_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_lb_body.position = (0, 1 - goal_clearance / 2)
        side_lb_shape = pymunk.Segment(side_lb_body, (0, -goal_clearance / 2), (0, goal_clearance / 2), 0.01)
        space.add(side_lb_body, side_lb_shape)

        side_rt_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_rt_body.position = (self.length, goal_clearance / 2)
        side_rt_shape = pymunk.Segment(side_rt_body, (0, -goal_clearance / 2), (0, goal_clearance / 2), 0.01)
        space.add(side_rt_body, side_rt_shape)

        side_rb_body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        side_rb_body.position = (self.length, 1 - goal_clearance / 2)
        side_rb_shape = pymunk.Segment(side_rb_body, (0, -goal_clearance / 2), (0, goal_clearance / 2), 0.01)
        space.add(side_rb_body, side_rb_shape)

        side_bodies = [goal0_body, goal1_body, side_top_body, side_bottom_body,
                       side_lt_body, side_lb_body, side_rt_body, side_rb_body]
        excl_side_bodies = [side_top_body, side_bottom_body,
                            side_lt_body, side_lb_body, side_rt_body, side_rb_body]
        # already have foo_bodies
        # already have ball_body

        for shape in space.shapes:
            shape.elasticity = 0.8

        # Set up collision masks
        side_filter = pymunk.ShapeFilter(1, 1, 2 | 4)
        foo_filter = pymunk.ShapeFilter(2, 2, 4)
        ball_filter = pymunk.ShapeFilter(4, 4, 1 | 2)
        goal_filter = pymunk.ShapeFilter(1, 1, 2)

        def apply_filter(bodies, shape_filter):
            for body in bodies:
                for shape in body.shapes:
                    shape.filter = shape_filter

        apply_filter(side_bodies, side_filter)
        apply_filter(rod_bodies, foo_filter)
        apply_filter([ball_body], ball_filter)
        apply_filter([goal0_body, goal1_body], goal_filter)

        for body in space.bodies:
            for shape in body.shapes: #type: pymunk.Shape
                shape.friction = 0

        return space, {
            "ball": ball_body,
            "rods": rod_bodies,
            "goals": [goal0_body, goal1_body],
            "excl_sides": excl_side_bodies
        }

    def get_rod_x(self, rod_idx, angle):
        start_x = self.rods[rod_idx][1]
        foo_h = self.foosman_size[1]

        return start_x - math.sin(angle * (math.pi / 2)) * foo_h

    def get_rod_angle(self, rod_idx, rod_x):
        start_x = self.rods[rod_idx][1]
        foo_h = self.foosman_size[1]

        asin_arg = (start_x - rod_x) / foo_h

        return math.asin(max(min(asin_arg, 1), -1)) / (math.pi / 2)

    def get_rod_offset(self, rod_idx, rod_y):
        # y is 0.5 +- 0.5 * self.rods[rod_idx].max_offset ([4])
        delta_y = rod_y - 0.5
        max_offset = self.rods[rod_idx][4]
        unclamped = (delta_y + max_offset / 2) / max_offset
        # print("unclamped", unclamped)

        return unclamped
        # return max(min(unclamped, 1), 0)

    def get_goal(self, position):
        x, y = position.real, position.imag
        if abs(y - 0.5) > self.goal_width / 2:
            return None
        if x < 0:
            return 0
        if x > self.length:
            return 1
        return None

    def get_inbounds(self, position, threshold=0.1):
        x, y = position.real, position.imag
        return -threshold <= x <= self.length + threshold and -threshold <= y <= 1 + threshold