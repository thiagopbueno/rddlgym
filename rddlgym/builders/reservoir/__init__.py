import numpy as np

from rddlgym.builders import *


class ReservoirBuilder(RDDLBuilder):

    MAX_RES_CAP = 100.0

    REQUIREMENTS = """
    requirements = {
        concurrent,
        reward-deterministic,
        intermediate-nodes,
        constrained-state
    };"""

    TYPES = """
    types {
        res: object;
    };"""

    NONFLUENTS = """MAX_RES_CAP(res): { non-fluent, real, default = 100.0 }; // Beyond this amount, water spills over

        UPPER_BOUND(res): { non-fluent, real, default = 80.0 };  // The upper bound for a safe reservoir level
        LOWER_BOUND(res): { non-fluent, real, default = 20.0 };  // The lower bound for a safe reservoir level

        RAIN_SHAPE(res):  { non-fluent, real, default = 25.0 };  // Gamma shape parameter for rainfall
        RAIN_SCALE(res):  { non-fluent, real, default = 25.0 };  // Gamma scale paramater for rainfall

        DOWNSTREAM(res,res): { non-fluent, bool, default = false }; // Indicates 2nd res is downstream of 1st res

        LOW_PENALTY(res) : { non-fluent, real, default =  -5.0 };     // Penalty per unit of level < LOWER_BOUND
        HIGH_PENALTY(res): { non-fluent, real, default = -100.0 };    // Penalty per unit of level > UPPER_BOUND
        SET_POINT_PENALTY(res): { non-fluent, real, default = -0.1 }; // Penalty per unit of level far from (LOWER_BOUND + UPPER_BOUND) / 2"""

    STATEFLUENTS = """
        rlevel(res): { state-fluent, real, default = 50.0 }; // Reservoir level for res"""

    INTERMFLUENTS = """
        vaporated(res): {interm-fluent, real, level=1}; // How much evaporates from res in this time step?
        rainfall(res):  {interm-fluent, real, level=1}; // How much rainfall is there in this time step?
        inflow(res):    {interm-fluent, real, level=1}; // How much res receives from upstream reservoirs?"""

    ACTIONFLUENTS = """
        outflow(res): { action-fluent, real, default = 0.0 }; // Action to set outflow of res"""

    INTERMCPFS = """vaporated(?r) = 1 / 2 * sin[rlevel(?r) / MAX_RES_CAP(?r)] * rlevel(?r);
        rainfall(?r) = Gamma(RAIN_SHAPE(?r), RAIN_SCALE(?r));
        inflow(?r) = sum_{?up : res} [DOWNSTREAM(?up,?r) * outflow(?up)];"""

    STATECPFS = """rlevel'(?r) = max[0.0, rlevel(?r) + rainfall(?r) - vaporated(?r) - (outflow(?r) * rlevel(?r)) + inflow(?r)];"""

    REWARD = """sum_{?r: res} [
        LOW_PENALTY * max[0.0, LOWER_BOUND(?r) - rlevel(?r)]
        + HIGH_PENALTY * max[0.0, rlevel(?r) - UPPER_BOUND(?r)]
        + SET_POINT_PENALTY * abs[(LOWER_BOUND(?r) + UPPER_BOUND(?r)) / 2 - rlevel(?r)]
    ];"""

        inflow(?r) = sum_{?up : res} [DOWNSTREAM(?up,?r)*(outflow(?up) + overflow(?up))];"""

    STATECPFS = """rlevel'(?r) = max[0.0, rlevel(?r) + rainfall(?r) - evaporated(?r) - outflow(?r) - overflow(?r) + inflow(?r)];"""

    REWARD = """sum_{?r: res} [if (rlevel'(?r)>=LOWER_BOUND(?r) ^ (rlevel'(?r)<=UPPER_BOUND(?r)))
                                    then 0
                                    else if (rlevel'(?r)<=LOWER_BOUND(?r))
                                        then LOW_PENALTY(?r)*(LOWER_BOUND(?r)-rlevel'(?r))
                                        else HIGH_PENALTY(?r)*(rlevel'(?r)-UPPER_BOUND(?r))];"""

    ACTIONPRECONDITIONS = """forall_{?r : res} outflow(?r) <= rlevel(?r);
        forall_{?r : res} outflow(?r) >= 0;"""

    STATEINVARIANTS = """forall_{?r : res} rlevel(?r) >= 0;
        forall_{?up : res} (sum_{?down : res} DOWNSTREAM(?up,?down)) <= 1;"""

    def __init__(self,
                 domain_id="reservoir",
                 non_fluents_id=None,
                 instance_id=None,
                 n_reservoirs=10,
                 level_set_point=0.5,
                 level_nominal_range=0.25,
                 rain_mean=0.20,
                 rain_variance=0.25,
                 init_relative_level=-0.45,
                 horizon=40,
                 discount=1.0):

        super().__init__()

        self.domain_id = domain_id
        self.non_fluents_id = non_fluents_id
        self.instance_id = instance_id

        self.n_reservoirs = n_reservoirs
        self.level_set_point = level_set_point
        self.level_nominal_range = level_nominal_range

        self.rain_mean = rain_mean * self.MAX_RES_CAP
        self.rain_variance = rain_variance * self.MAX_RES_CAP

        self.init_relative_level = init_relative_level

        self.horizon = horizon
        self.discount = discount

    @property
    def _objects(self):
        objects = generate_objs_list("t", self.n_reservoirs)
        return f"res : {{ {objects} }};"

    @property
    def _non_fluents(self):
        RAIN_SHAPE = generate_predicate_list("RAIN_SHAPE", "t", self.rain_shape)
        RAIN_SCALE = generate_predicate_list("RAIN_SCALE", "t", self.rain_scale)

        DOWNSTREAM = generate_topology("DOWNSTREAM", "t", self.downstream)
        LOWER_BOUND = generate_predicate_list("LOWER_BOUND", "t", self.lower_bound)
        UPPER_BOUND = generate_predicate_list("UPPER_BOUND", "t", self.upper_bound)

        return f"""{DOWNSTREAM}

        {LOWER_BOUND}

        {UPPER_BOUND}

        {RAIN_SHAPE}

        {RAIN_SCALE}"""

    @property
    def initial_state(self):
        return [self.MAX_RES_CAP * (self.level_set_point + self.init_relative_level)] * self.n_reservoirs

    @property
    def _init_state(self):
        return generate_predicate_list("rlevel", "t", self.initial_state)

    @property
    @config
    def downstream(self):
        downstream = np.zeros([self.n_reservoirs] * 2, dtype=np.int32)
        for i in range(self.n_reservoirs - 1):
            downstream[i, i + 1] = 1
        return downstream.tolist()

    @property
    @config
    def lower_bound(self):
        return [self.MAX_RES_CAP * max(0.0, self.level_set_point - self.level_nominal_range / 2)] * self.n_reservoirs

    @property
    @config
    def upper_bound(self):
        return [self.MAX_RES_CAP * min(1.0, self.level_set_point + self.level_nominal_range / 2)] * self.n_reservoirs

    @property
    @config
    def rain_shape(self):
        return [(self.rain_mean ** 2) / self.rain_variance] * self.n_reservoirs

    @property
    @config
    def rain_scale(self):
        return [self.rain_variance / self.rain_mean] * self.n_reservoirs


if __name__ == "__main__":
    builder = ReservoirBuilder(
        "reservoir",
        non_fluents_id="res10",
        instance_id="inst_reservoir_res10",
        n_reservoirs=10,
        level_set_point=0.5,
        level_nominal_range=0.25,
        rain_mean=0.20,
        rain_variance=0.25,
        init_relative_level=-0.45
    )

    rddl = builder.build()
    print(rddl)

    builder.save("res10.rddl")
    builder.dump_config("res10.config.json")

