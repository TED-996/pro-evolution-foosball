from . import table
import pymunk
import numpy as np
import math


class Simulation:
    """
    Has:
        A TableInfo
        A State
    """

    def __init__(self, table_info: table.TableInfo):
        self.table_info = table_info
        self.state = self.table_info.get_init_state()

        self.space: pymunk.Space
        self.rod_bodies: [pymunk.Body]
        self.ball_body: pymunk.Body
        self.goal_bodies: (pymunk.Body, pymunk.Body)

        self.space, bodies = self.table_info.get_space()
        self.rod_bodies = bodies["rods"]
        self.ball_body = bodies["ball"]
        self.goal_bodies = tuple(bodies["goals"])
        self.side_bodies = tuple(bodies["excl_sides"])

        self.rod_body_idx_cache = _inverse_list(self.rod_bodies, key=id)
        self._hook_velocities()

    # noinspection PyUnusedLocal
    def _get_rod_velocity(self, body: pymunk.Body, gravity, damping, dt):
        rod_idx = self.rod_body_idx_cache[id(body)]
        rod = self.state.rods[rod_idx]
        offset, offset_vel = rod[0]
        angle, angle_vel = rod[1]
        max_offset = self.table_info.rods[rod_idx][4]

        next_foo_x = self.table_info.get_rod_x(rod_idx, angle + angle_vel * dt)
        vel_x = (next_foo_x - body.position[0]) / dt

        a_offset = self.table_info.get_rod_offset(rod_idx, body.position[1])
        if a_offset < 0 and offset_vel < 0:
            offset_vel = 0
        if a_offset > max_offset and offset_vel > 0:
            offset_vel = 0

        vel_y = offset_vel
        # vel_y = 0

        # return vel_x, vel_y
        body.velocity = (vel_x, vel_y)

    def _hook_velocities(self):
        for rod in self.rod_bodies:
            rod.velocity_func = self._get_rod_velocity

    def _fetch_state(self):
        ball_offset = self.ball_body.position
        ball_velocity = self.ball_body.velocity
        self.state.ball = (complex(ball_offset[0], ball_offset[1]),
                           complex(ball_velocity[0], ball_velocity[1]))
        for idx, body in enumerate(self.rod_bodies):
            rod_offset = self.table_info.get_rod_offset(idx, body.position[1])
            rod_angle = self.table_info.get_rod_angle(idx, body.position[0])
            rod_last_angle = self.table_info.get_rod_angle(idx, body.position[0] - body.velocity[0])
            self.state.rods[idx] = ((rod_offset, body.velocity[1]),
                                    (rod_angle, rod_angle - rod_last_angle))

    def apply_inputs(self, side, input):
        self.state.apply_inputs(self._input_to_absolute(side, input))

    def _input_to_absolute(self, side, input):
        rod_idx, offset_vel, angle_vel = input

        # Side 1's rods are ordered in reverse
        if side == 0:
            ordered = self.table_info.rods
            idxs = range(len(ordered))
        else:
            ordered = reversed(self.table_info.rods)
            idxs = range(len(self.table_info.rods) - 1, -1, -1)


        # Find the rod_idx-th rod in the player's order
        # The player's rods are those with side == rod[0]
        idx_left = rod_idx
        for abs_idx, rod in zip(idxs, ordered):
            if side == rod[0]:
                if idx_left == 0:
                    return abs_idx, offset_vel, angle_vel
                idx_left -= 1

        raise ValueError("rod_idx {} too large".format(rod_idx))

    def tick(self, time):
        # _assert_no_nans(self.space)

        self.space.step(time)
        self._fetch_state()

        # _assert_no_nans(self.space)

    def get_current_reward(self, player: int):
        """
        :param player: for what player to compute the score;
                       must be:
                        0 for player with the goal at x = 0 coordinate
                        1 for player with the goal at x = MAX_X coordinate)
        :return: a score for current state for player player
        """
        # to be deleted after integration
        assert isinstance(player, int), "Player must be a integer"
        half_goal = self.table_info.goal_width / 2
        player_starting_point = player * self.table_info.length  # i.e X coordinate for goal
        # check for goal
        if player_starting_point - (((-1) ** (1 ^ player)) * self.state.ball[0].real) <= 0 \
                and ((0.5 - half_goal) < self.state.ball[0].imag < (0.5 + half_goal)):
            return 100
        ball_direction = np.sign(self.state.ball[1].real) * ((-1) ** player)
        if ball_direction == 0:
            return -1  # penalty for inert state of the ball

        # return a score that is a sum of:
        #   how far is the ball from goal of player
        #   50% of above number (negative weighted if player doesn't have possession, positive otherwise )
        return abs(player_starting_point - self.state.ball[0].real) + \
            abs(player_starting_point - self.state.ball[0].real) / 2 * ball_direction

    def get_rod_owners(self):
        return [r[0] for r in self.table_info.rods]


def _inverse_list(items, key):
    results = {}
    for idx, item in enumerate(items):
        results[key(item)] = idx

    return results


def _assert_no_nans(space: pymunk.Space):
    def any_nan(t):
        return any(math.isnan(x) for x in t)

    for body in space.bodies:  # type:pymunk.Body
        if any_nan(body.position) or any_nan(body.velocity):
            raise ValueError("NaN in body!")
        for shape in body.shapes:  # type:pymunk.Shape
            if any_nan((shape.bb.top, shape.bb.bottom, shape.bb.left, shape.bb.right))\
                    or any_nan((shape.mass,)) or any_nan((shape.moment,)):
                raise ValueError("NaN in shape!")
