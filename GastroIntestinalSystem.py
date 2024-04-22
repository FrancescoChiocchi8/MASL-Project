import math
import sys
import random as rd
from repast4py import core, space, schedule, logging, random
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
import numpy as np
from typing import Tuple, Dict
from repast4py import context as ctx
from repast4py import space
from dataclasses import dataclass
from mpi4py import MPI
from numba import int32
from numba.experimental import jitclass
from repast4py import parameters


model = None

spec = [
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),    
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]

#it take a location and add an array of offsets to that location to create a new array consisting of the neighboring coordinates
@jitclass(spec)
class GridNghFinder: 

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y
        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]
        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


class SCFA(core.Agent): 

    TYPE = 0 

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=SCFA.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        grid = model.microibiotaGrid
        pt = grid.get_location(self)
    

class LPS(core.Agent):

    TYPE = 1

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = LPS.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def stepMicrobiota(self):
        grid = model.microibiotaGrid
        pt = grid.get_location(self)
        #model.moveToLume(self)            
    def stepLume(self):
        grid = model.microibiotaGrid
        pt = grid.get_location(self)


class CellulaEpiteliale(core.Agent):

    TYPE = 2


    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = CellulaEpiteliale.TYPE, rank=rank)
        self.permeability = 10

    def save(self) -> Tuple:
        return (self.uid, self.permeability)
    
    def getPermeability(self):
        return self.permeability
    
    def getNumberOfSCFA(self):
       countSCFA = 0
       for s in model.MicrobiotaContext.agents(SCFA.TYPE):
           countSCFA += 1
       
       return countSCFA

    """
    def min_SCFA(self):
        scfaTotal = []
        for i in model.MicrobiotaContext.agents(SCFA.TYPE):
            scfaTotal.append(i)
        
        min_scfa = (len(scfaTotal) * 75) / 100
        return min_scfa
    """

    def step(self):
        num_SCFA = self.getNumberOfSCFA()
        if num_SCFA <= 375: #self.min_SCFA():
            self.permeability += (self.permeability * 15) / 100


"""         
class AlfaSinucleina(core.Agent):

    TYPE = 3

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = AlfaSinucleina.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
"""

#classe TNF-alfa che ha un una variabile booleana "rispostaImmunitaria" che quando gli LPS saranno un numero elevato
#verrà messa a True. Questo rappresenta lo stato di infiammazione del sistema che porta alla produzione di alpha sinucleina all'intero
#del sistema gastro intestinale


agent_cache = {} 

def restore_agent(agent_data: Tuple): 
    #uid element 0 is id, 1 is type, 2 is rank
    uid = agent_data[0]                                         
    
    if uid[1] == SCFA.TYPE:                                      
        if uid in agent_cache:                                  
            return agent_cache[uid]
        else:
            s = SCFA(uid[0], uid[2])
            agent_cache[uid] = s
            return s
        
    if uid[1] == LPS.TYPE:                                                       
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            l = LPS(uid[0], uid[2])
            agent_cache[uid] = l
            return l
    
    if uid[1] == CellulaEpiteliale.TYPE:                                                       
        if uid in agent_cache:
            c = agent_cache[uid]
        else:
            c = CellulaEpiteliale(uid[0], uid[2])
            agent_cache[uid] = c
        
        c.permeability = agent_data[1]
        return c



@dataclass
class MicrobiotaCounts:
    scfa: int = 0
    lps: int = 0
    permeability: float = 0.0
    cellEpit: int = 0

@dataclass
class LumeCounts:
    lps: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.MicrobiotaContext = ctx.SharedContext(comm)
        self.LumeContext = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_endMicrobiota)
        self.runner.schedule_end_event(self.at_endLume)

        # microbiota world
        box1 = space.BoundingBox(0, params['microbiotaWorld.width'], 0, params['microbiotaWorld.height'], 0, 0)    
        self.microibiotaGrid = space.SharedGrid('grid', bounds=box1, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)    
        self.MicrobiotaContext.add_projection(self.microibiotaGrid)

        self.microbiotaSpace = space.SharedCSpace('space', bounds=box1, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)    
        self.MicrobiotaContext.add_projection(self.microbiotaSpace)

        # lume world
        box2 = space.BoundingBox(0, params['lumeWorld.width'], 0, params['lumeWorld.height'], 0, 0)    
        self.lumeGrid = space.SharedGrid('grid', bounds=box2, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)    
        self.LumeContext.add_projection(self.lumeGrid)

        self.lumeSpace = space.SharedCSpace('space', bounds=box2, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)    
        self.LumeContext.add_projection(self.lumeSpace)


        #self.ngh_finder = GridNghFinder(0, 0, box1.xextent, box1.yextent)
        #self.ngh_finder = GridNghFinder(0, 0, box2.xextent, box2.yextent)

        #logging
        self.microbiotaCounts = MicrobiotaCounts()    
        microbiotaLoggers = logging.create_loggers(self.microbiotaCounts, op=MPI.SUM, rank=self.rank)    
        self.microbiotaData_set = logging.ReducingDataSet(microbiotaLoggers, self.comm, params['logging_file']) 

        self.lumeCounts = LumeCounts()    
        lumeLoggers = logging.create_loggers(self.lumeCounts, op=MPI.SUM, rank=self.rank)    
        self.lumeData_set = logging.ReducingDataSet(lumeLoggers, self.comm, params['lumeLogging_file']) 


        world_size = comm.Get_size()

        #add scfa agents to microbiota context
        total_scfa_count = params['scfa.count']    
        pp_scfa_count = int(total_scfa_count / world_size)   #number of scfa per processor 
        if self.rank < total_scfa_count % world_size:    
            pp_scfa_count += 1

        local_bounds = self.microbiotaSpace.get_local_bounds()    
        for i in range(pp_scfa_count):    
            s = SCFA(i, self.rank)    
            self.MicrobiotaContext.add(s)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(s, x, y) 


        #add lps agents to microbiota context
        total_lps_count = params['lps.count']    
        pp_lps_count = int(total_lps_count / world_size)   #number of lps per processor 
        if self.rank < total_lps_count % world_size:    
            pp_lps_count += 1

        local_bounds = self.microbiotaSpace.get_local_bounds()    
        for i in range(pp_lps_count):    
            l = LPS(i, self.rank)    
            self.MicrobiotaContext.add(l)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(l, x, y) 


        #add epitelial cell to microbiota context    
        total_epitCell_count = params['epitelialCell.count']
        pp_epitCell_count = int(total_epitCell_count / world_size)
        if self.rank < total_epitCell_count % world_size:
            pp_epitCell_count += 1
        
        local_bounds = self.microbiotaSpace.get_local_bounds()  
        for i in range(pp_epitCell_count):    
            c = CellulaEpiteliale(i, self.rank)    
            self.MicrobiotaContext.add(c)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(c, x, y)


        #add lps to lume context
        total_lpsLume_count = params['lpslume.count']    
        pp_lpsLume_count = int(total_lpsLume_count / world_size)   #number of lps per processor 
        if self.rank < total_lpsLume_count % world_size:    
            pp_lpsLume_count += 1

        local_boundsLume = self.lumeSpace.get_local_bounds()    
        for i in range(pp_lpsLume_count):    
            p = LPS(i, self.rank)    
            self.LumeContext.add(p)    
            x = random.default_rng.uniform(local_boundsLume.xmin, local_boundsLume.xmin + local_boundsLume.xextent)    
            y = random.default_rng.uniform(local_boundsLume.ymin, local_boundsLume.ymin + local_boundsLume.yextent)
            self.move(p, x, y)

        
    def at_endMicrobiota(self):
        self.microbiotaData_set.close()
    
    def at_endLume(self):
        self.lumeData_set.close()

    def move(self, agent, x, y):
        self.microbiotaSpace.move(agent, cpt(x, y))
        self.microibiotaGrid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def moveToLume(self, agent):
        self.MicrobiotaContext.remove(agent)
        self.LumeContext.add(agent)

    def step(self):
        tick = self.runner.schedule.tick    
        self.log_countsMicrobiota(tick) 
        self.log_countsLume(tick)   
        self.MicrobiotaContext.synchronize(restore_agent) 

        for s in self.MicrobiotaContext.agents(SCFA.TYPE):
            s.step()
   
        for l in self.MicrobiotaContext.agents(LPS.TYPE):  
            if self.permeability() >= 60:
                self.moveToLume(l)
                l.step()
        
        for c in self.MicrobiotaContext.agents(CellulaEpiteliale.TYPE):  
            c.step()
        
        if tick >= 10:
            self.removeSCFA() 

    def permeability(self):
        for ce in self.MicrobiotaContext.agents(CellulaEpiteliale.TYPE):
            permeability = ce.getPermeability()

        print(permeability)
        return permeability


    def removeSCFA(self):
        num_SCFA = []
        for i in self.MicrobiotaContext.agents(SCFA.TYPE):
            num_SCFA.append(i)
            
        countSCFA = len(num_SCFA)
        if countSCFA > 0:
            randomPosition = rd.randint(0, countSCFA - 1)
            self.MicrobiotaContext.remove(num_SCFA[randomPosition])
   
    def run(self):
        self.runner.execute()
    

    def log_countsMicrobiota(self, tick):
        num_MicrobiotaAgents = self.MicrobiotaContext.size([SCFA.TYPE, LPS.TYPE, CellulaEpiteliale.TYPE]) 

        self.microbiotaCounts.scfa = num_MicrobiotaAgents[SCFA.TYPE]    
        self.microbiotaCounts.lps = num_MicrobiotaAgents[LPS.TYPE]
        self.microbiotaCounts.cellEpit = num_MicrobiotaAgents[CellulaEpiteliale.TYPE]
        self.microbiotaCounts.permeability = self.permeability() 

        self.microbiotaData_set.log(tick)
    

    def log_countsLume(self, tick):
        num_LumeAgents= self.LumeContext.size([LPS.TYPE])

        self.lumeCounts.lps = num_LumeAgents[LPS.TYPE]
        self.lumeData_set.log(tick)



def run(params: Dict):
    global model    
    model = Model(MPI.COMM_WORLD, params)   
    model.run()

if __name__ == "__main__":
    parser = parameters.create_args_parser()    
    args = parser.parse_args()   
    params = parameters.init_params(args.parameters_file, args.parameters)    
    run(params)