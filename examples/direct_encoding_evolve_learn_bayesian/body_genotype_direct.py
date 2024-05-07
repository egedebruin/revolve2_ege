import copy
import uuid
from abc import abstractmethod

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
    children: dict
    possible_children: list
    type: str
    body_module = None

    def __init__(self, rotation):
        self.rotation = rotation
        self.children = {}

    def develop(self, body, grid=None):
        if grid is None:
            grid = []
        for direction, module in self.children.items():
            new_module = module.get_body_module()
            setattr(self.body_module, direction, new_module)
            module.develop(body, grid)

            grid_position = body.grid_position(new_module)
            if grid_position not in grid:
                grid.append(grid_position)
            else:
                setattr(self.body_module, direction, None)
                continue

    def get_body_module(self):
        pass

    def add_random_module(self, rng: np.random.Generator, brain: BrainGenotype):
        direction_chooser = rng.choice(self.possible_children)

        if direction_chooser in self.children:
            self.children[direction_chooser].add_random_module(rng, brain)
        else:
            self.children[direction_chooser] = self.choose_random_module(rng, brain)

    def add_random_module_to_connection(self, index: int, rng: np.random.Generator, brain: BrainGenotype):
        for module in self.possible_children:
            if module in self.children.keys():
                index = self.children[module].add_random_module_to_connection(index, rng, brain)
                if index == 0:
                    return 0
            else:
                if index == 1:
                    self.children[module] = self.choose_random_module(rng, brain)
                    return 0
                index -= 1
        return index

    def get_amount_possible_connections(self):
        possible_connections = 0
        for module in self.possible_children:
            if module in self.children.keys():
                possible_connections += self.children[module].get_amount_possible_connections()
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

    def serialize(self):
        serialized = {'type': self.type, 'rotation': self.rotation, 'children': {}}

        for direction, module in self.children.items():
            serialized['children'][direction] = module.serialize()

        return serialized

    def deserialize(self, serialized):
        self.type = serialized['type']
        for direction, child in serialized['children'].items():
            if child['type'] == 'brick':
                child_object = BrickGenotype(child['rotation'])
            else:
                child_object = HingeGenotype(child['rotation'])
            self.children[direction] = child_object.deserialize(child)

        return self

    def choose_random_module(self, rng: np.random.Generator, brain: BrainGenotype):
        module_chooser = rng.random()
        rotation = rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value, RightAngles.DEG_270.value])

        if module_chooser < 0.5:
            module = BrickGenotype(rotation)
        else:
            module = HingeGenotype(rotation)

            new_brain_chooser = rng.random()
            if config.CONTROLLERS == -1 and new_brain_chooser < config.NEW_HINGE_NEW_BRAIN:
                module.brain_index = brain.add_new()
            else:
                module.brain_index = rng.choice(list(brain.brain.keys()))

        return module


class CoreGenotype(ModuleGenotype):
    possible_children = ['left', 'right', 'front', 'back', 'down', 'up']
    type = 'core'
    rotation = 0.0

    def develop(self, body, grid=None):
        self.body_module = body.core_v1
        super().develop(body)

    def get_amount_nodes(self):
        nodes = 0

        for module in self.children.values():
            nodes += module.get_amount_nodes()

        return nodes


class BrickGenotype(ModuleGenotype):
    possible_children = ['left', 'right', 'front', 'up', 'down']
    type = 'brick'

    def get_body_module(self):
        self.body_module = BrickV1(self.rotation)
        return self.body_module


class HingeGenotype(ModuleGenotype):
    possible_children = ['attachment']
    brain_index = -1
    type = 'hinge'

    def develop(self, body, grid=None):
        super().develop(body)
        self.body_module.map_uuid = self.brain_index

    def get_body_module(self):
        self.body_module = ActiveHingeV1(self.rotation)
        return self.body_module

    def check_for_brains(self):
        uuids = super().check_for_brains()
        if self.brain_index not in uuids:
            uuids.append(self.brain_index)

        return uuids

    def switch_brain(self, rng: np.random.Generator, brain: BrainGenotype):
        if rng.random() > config.SWITCH_BRAIN:
            self.brain_index = rng.choice(list(brain.brain.keys()))

        super().switch_brain(rng, brain)

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

    @classmethod
    def initialize_body(cls, rng: np.random.Generator, brain: BrainGenotype):
        number_of_modules = rng.integers(config.INIT_MIN_MODULES, config.INIT_MAX_MODULES)
        body = CoreGenotype(0.0)
        for _ in range(number_of_modules):
            amount_possible_connections = body.get_amount_possible_connections()
            connection_to_add = rng.integers(1, amount_possible_connections + 1)
            body.add_random_module_to_connection(connection_to_add, rng, brain)

        return BodyGenotypeDirect(body=body)

    def mutate_body(self, rng: np.random.Generator, brain: BrainGenotype):
        body = copy.deepcopy(self.body)

        mutation_chooser = rng.random()

        if mutation_chooser < 0.15:
            for _ in range(rng.integers(1, config.MAX_ADD_MODULES + 1)):
                amount_possible_connections = body.get_amount_possible_connections()
                connection_to_add = rng.integers(1, amount_possible_connections + 1)
                body.add_random_module_to_connection(connection_to_add, rng, brain)
        elif mutation_chooser < 0.30:
            for _ in range(rng.integers(1, config.MAX_DELETE_MODULES + 1)):
                amount_nodes = body.get_amount_nodes()
                if amount_nodes == 0:
                    break
                node_to_remove = rng.integers(1, amount_nodes + 1)
                body.remove_node(node_to_remove)

            if config.CONTROLLERS == -1:
                used_brains = body.check_for_brains()
                brain.remove_unused(used_brains)
        elif mutation_chooser < 0.45:
            body.switch_brain(rng, brain)

            if config.CONTROLLERS == -1:
                used_brains = body.check_for_brains()
                brain.remove_unused(used_brains)
        return BodyGenotypeDirect(body=body), mutation_chooser

    def get_brain_uuids(self):
        return self.body.check_for_brains()

    @staticmethod
    def crossover_body(parent1: 'BodyGenotypeDirect', parent2: 'BodyGenotypeDirect', rng: np.random.Generator):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if len(parent1.body.children) == 0 or len(parent2.body.children) == 0:
            return child1, child2

        parent_1_branch_chooser = rng.choice(list(parent1.body.children))
        parent_2_branch_chooser = rng.choice(list(parent2.body.children))

        child1.body.children[parent_1_branch_chooser] = copy.deepcopy(parent2.body.children[parent_2_branch_chooser])
        child2.body.children[parent_2_branch_chooser] = copy.deepcopy(parent1.body.children[parent_1_branch_chooser])

        child1.body.children[parent_1_branch_chooser].rotation = (
            rng.choice([RightAngles.DEG_0.value, RightAngles.DEG_90.value, RightAngles.DEG_180.value, RightAngles.DEG_270.value]))

        return child1, child2

    def develop_body(self):
        body = BodyV1()
        self.body.develop(body)
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
