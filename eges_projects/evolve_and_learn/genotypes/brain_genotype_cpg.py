import numpy as np
from cpg_brain import CpgBrain
from revolve2.modular_robot.body.v1 import BodyV1
from genotypes.brain_genotype import BrainGenotype as AbstractBrainGenotype


class BrainGenotype(AbstractBrainGenotype):

    def develop_brain(self, body: BodyV1):
        cpg_brain = CpgBrain(
            body,
            self.brain
        )
        return cpg_brain

    def get_p_bounds(self):
        brain_uuids = list(self.brain.keys())
        pbounds = {}
        for key in brain_uuids:
            pbounds['internal_' + str(key)] = [0, 1]
            pbounds['first_neighbour_' + str(key)] = [0, 1]
            pbounds['second_neighbour_' + str(key)] = [0, 1]
        return pbounds

    def get_evolutionary_search_next_point(self):
        brain_uuids = list(self.brain.keys())
        next_point = {}
        for key in brain_uuids:
            next_point['internal_' + str(key)] = self.brain[key][0]
            next_point['first_neighbour_' + str(key)] = self.brain[key][1]
            next_point['second_neighbour_' + str(key)] = self.brain[key][2]
        return next_point

    def next_point_to_brain(self, next_point, brain_uuids):
        for brain_uuid in brain_uuids:
            self.brain[brain_uuid] = np.array(
                [
                    next_point['internal_' + str(brain_uuid)],
                    next_point['first_neighbour_' + str(brain_uuid)],
                    next_point['second_neighbour_' + str(brain_uuid)],
                ]
            )
