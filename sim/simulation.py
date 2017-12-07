from . import table, input


class Simulation:
    """
    Has:
        A TableInfo
        A State
    """

    def __init__(self, table_info: table.TableInfo):
        self.table_info = table_info
        self.state = self.table_info.get_init_state()
        self.inputs_applied = []

    def apply_inputs(self, input:input.Input):
        assert len(self.inputs_applied) < 2 and input.side not in self.inputs_applied
        self.inputs_applied.append(input.side)

        self.state.apply_inputs(input)

    def tick(self, time):
        pass