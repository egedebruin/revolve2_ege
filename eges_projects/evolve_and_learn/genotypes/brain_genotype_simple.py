import math
import numpy as np
from sine_brain_simple import SineBrain
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.body.v1 import BodyV1
from genotypes.brain_genotype import BrainGenotype as AbstractBrainGenotype


class BrainGenotype(AbstractBrainGenotype):

    def develop_brain(self, body: BodyV1):
        active_hinges = body.find_modules_of_type(ActiveHinge)

        amplitudes = []
        phases = []
        offsets = []
        for active_hinge in active_hinges:
            amplitudes.append(self.brain[active_hinge.map_uuid][0])
            phases.append(self.brain[active_hinge.map_uuid][1] * 2 * math.pi)
            offsets.append(self.brain[active_hinge.map_uuid][2] - 0.5)

        brain = SineBrain(
            active_hinges=active_hinges,
            amplitudes=amplitudes,
            phases=phases,
            offsets=offsets
        )

        return brain

    def get_p_bounds(self):
        brain_uuids = list(self.brain.keys())
        pbounds = {}
        for key in brain_uuids:
            pbounds['amplitude_' + str(key)] = [0, 1]
            pbounds['phase_' + str(key)] = [0, 1]
            pbounds['offset_' + str(key)] = [0, 1]
        return pbounds

    def get_evolutionary_search_next_point(self):
        brain_uuids = list(self.brain.keys())
        next_point = {}
        for key in brain_uuids:
            next_point['amplitude_' + str(key)] = self.brain[key][0]
            next_point['phase_' + str(key)] = self.brain[key][1]
            next_point['offset_' + str(key)] = self.brain[key][2]
        return next_point

    def next_point_to_brain(self, next_point, brain_uuids):
        for brain_uuid in brain_uuids:
            self.brain[brain_uuid] = np.array(
                [
                    next_point['amplitude_' + str(brain_uuid)],
                    next_point['phase_' + str(brain_uuid)],
                    next_point['offset_' + str(brain_uuid)],
                ]
            )
