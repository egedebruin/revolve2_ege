import copy
import math
import uuid
from pyrr import Vector3

import numpy as np
import sqlalchemy.orm as orm
import json
from sqlalchemy import event
from sqlalchemy.engine import Connection

from brain_genotype import BrainGenotype
import config
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import BodyV1, ActiveHingeV1, BrickV1


class ModuleGenotype:
    rotation: float
    temp_rotation: float
    children: dict
    possible_children: list
    type: str
    body_module = None
    reverse_rotation = {
        RightAngles.DEG_0.value: 0.0,
        RightAngles.DEG_90.value: RightAngles.DEG_270.value,
        RightAngles.DEG_180.value: RightAngles.DEG_180.value,
        RightAngles.DEG_270.value: RightAngles.DEG_90.value
    }
    reverse_direction = {
        'left': 'right',
        'up': 'up',
        'front': 'front',
        'right': 'left',
        'down': 'down',
        'back': 'back',
        'attachment': 'attachment'
    }

    def __init__(self, rotation):
        self.rotation = rotation
        self.children = {}

    def develop(self, body, grid, mirror):
        if grid is None:
            grid = [Vector3([0, 0, 0])]
        for directions, module in self.children.items():
            if isinstance(directions, str):
                directions = [directions]
            is_mirror = mirror
            for direction in directions:
                new_module = module.get_body_module(is_mirror)
                if mirror:
                    direction = self.reverse_direction[direction]
                setattr(self.body_module, direction, new_module)
                module.develop(body, grid, is_mirror)

                grid_position = body.grid_position(new_module)
                if grid_position not in grid:
                    grid.append(grid_position)
                else:
                    setattr(self.body_module, direction, None)
                    continue

                is_mirror = not is_mirror

    def get_body_module(self, reverse):
        self.temp_rotation = self.rotation
        if reverse:
            self.temp_rotation = self.reverse_rotation[self.rotation]

    def add_random_module_to_connection(self, index: int, rng: np.random.Generator, brain: BrainGenotype):
        for directions in self.possible_children:
            if tuple(directions) in self.children.keys():
                index = self.children[tuple(directions)].add_random_module_to_connection(index, rng, brain)
            else:
                if index == 1:
                    self.children[tuple(directions)] = self.choose_random_module(rng, brain)
                index -= 1
            if index == 0:
                return 0
        return index

    def get_amount_possible_connections(self):
        possible_connections = 0
        for directions in self.possible_children:
            if tuple(directions) in self.children.keys():
                possible_connections += self.children[tuple(directions)].get_amount_possible_connections()
            else:
                possible_connections += 1
        return possible_connections

    def get_amount_leaf_nodes(self):
        leaves = 0

        for module in self.children.values():
            leaves += module.get_amount_leaf_nodes()

        if leaves == 0:
            leaves = 1
        return leaves

    def get_amount_nodes(self):
        nodes = 1

        for module in self.children.values():
            nodes += module.get_amount_nodes()

        return nodes

    def get_amount_hinges(self):
        nodes = 0

        for module in self.children.values():
            nodes += module.get_amount_hinges()

        return nodes

    def is_leaf_node(self):
        return not bool(self.children)

    def remove_leaf_node(self, index):
        for direction, module in self.children.items():
            if module.is_leaf_node():
                if index == 1:
                    self.children.pop(direction)
                    return 0
                index -= 1
            else:
                index = module.remove_leaf_node(index)
                if index == 0:
                    return 0

        return index

    def remove_node(self, index):
        for direction, module in self.children.items():
            if index == 1:
                self.children.pop(direction)
                return 0
            index = module.remove_node(index - 1)
            if index == 0:
                return 0
        return index

    def check_for_brains(self):
        uuids = []
        for module in self.children.values():
            recursive_uuids = module.check_for_brains()
            for recursive_uuid in recursive_uuids:
                if recursive_uuid not in uuids:
                    uuids.append(recursive_uuid)
        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        for module in self.children.values():
            module.switch_brain(rng, brain)

    def reverse_phase_function(self, value):
        for module in self.children.values():
            module.reverse_phase_function(value)

    def serialize(self):
        serialized = {'type': self.type, 'rotation': self.rotation, 'children': {}}

        for directions, module in self.children.items():
            direction_string = ",".join(directions)
            serialized['children'][direction_string] = module.serialize()

        return serialized

    def deserialize(self, serialized):
        self.type = serialized['type']
        for direction, child in serialized['children'].items():
            if isinstance(direction, str):
                direction = tuple(map(str, direction.split(',')))
            if child['type'] == 'brick':
                child_object = BrickGenotype(child['rotation'])
            else:
                child_object = HingeGenotype(child['rotation'])
            self.children[direction] = child_object.deserialize(child)

        return self


class CoreGenotype(ModuleGenotype):
    possible_children = [['left'], ['right'], ['front', 'back']]
    type = 'core'
    rotation = 0.0
    possible_phase_differences = [0, math.pi]
    reverse_phase = 0

    def develop(self, body, grid, mirror):
        self.body_module = body.core_v1
        super().develop(body, grid, mirror)

    def get_amount_nodes(self):
        nodes = 0

        for module in self.children.values():
            nodes += module.get_amount_nodes()

        return nodes

    def serialize(self):
        serialized = super().serialize()

        phase_difference_to_value = {
            0: 0,
            0.5 * math.pi: 2,
            math.pi: 1,
            1.5 * math.pi: 3,
        }

        serialized['reverse_phase'] = phase_difference_to_value[self.reverse_phase]

        return serialized

    def deserialize(self, serialized):
        super().deserialize(serialized)

        value_to_phase_difference = {
            0: 0,
            2: 0.5 * math.pi,
            1: math.pi,
            3: 1.5 * math.pi,
        }

        self.reverse_phase = value_to_phase_difference[serialized['reverse_phase']]

        return self


class BrickGenotype(ModuleGenotype):
    possible_children = [['left'], ['right'], ['front'], ['up'], ['down']]
    type = 'brick'

    def get_body_module(self, reverse):
        super().get_body_module(reverse)
        self.body_module = BrickV1(self.temp_rotation)
        return self.body_module


class HingeGenotype(ModuleGenotype):
    possible_children = [['attachment']]
    brain_index = -1
    reverse_phase_value = 0
    type = 'hinge'

    def develop(self, body, grid, mirror):
        super().develop(body, grid, mirror)
        self.body_module.map_uuid = self.brain_index

        if mirror:
            self.body_module.reverse_phase = self.reverse_phase_value
        else:
            self.body_module.reverse_phase = 0

    def get_body_module(self, reverse):
        super().get_body_module(reverse)
        self.body_module = ActiveHingeV1(self.temp_rotation)
        return self.body_module

    def check_for_brains(self):
        uuids = super().check_for_brains()
        if self.brain_index not in uuids:
            uuids.append(self.brain_index)

        return uuids

    def reverse_phase_function(self, value):
        self.reverse_phase_value = value

        super().reverse_phase_function(value)

    def serialize(self):
        serialized = super().serialize()
        serialized['brain_index'] = str(self.brain_index)

        return serialized

    def deserialize(self, serialized):
        super().deserialize(serialized)
        self.brain_index = uuid.UUID(serialized['brain_index'])

        return self

    def get_amount_hinges(self):
        nodes = 1

        for module in self.children.values():
            nodes += module.get_amount_hinges()

        return nodes


class BodyGenotypeDirect(orm.MappedAsDataclass):
    """SQLAlchemy model for a direct encoding body genotype."""

    body: CoreGenotype

    _serialized_body: orm.Mapped[str] = orm.mapped_column(
        "serialized_body", init=False, nullable=False
    )

    def __init__(self, body: CoreGenotype):
        self.body = body

    def get_brain_uuids(self):
        return self.body.check_for_brains()

    def develop_body(self):
        body = BodyV1()
        self.body.reverse_phase_function(self.body.reverse_phase)
        self.body.develop(body, None, False)
        return body


@event.listens_for(BodyGenotypeDirect, "before_update", propagate=True)
@event.listens_for(BodyGenotypeDirect, "before_insert", propagate=True)
def _update_serialized_body(
        mapper: orm.Mapper[BodyGenotypeDirect],
        connection: Connection,
        target: BodyGenotypeDirect,
) -> None:
    target._serialized_body = str(target.body.serialize()).replace("'", "\"")


@event.listens_for(BodyGenotypeDirect, "load", propagate=True)
def _deserialize_body(target: BodyGenotypeDirect, context: orm.QueryContext) -> None:
    serialized_dict = json.loads(target._serialized_body)
    target.body = CoreGenotype(0.0).deserialize(serialized_dict)
