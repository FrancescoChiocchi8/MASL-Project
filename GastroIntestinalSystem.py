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
        nghs = model.ngh_finderMicrobiota.find(pt.x, pt.y)

        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == CellulaEpiteliale.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            space_pt = model.microbiotaSpace.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.5
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
            
        #pt = grid.get_location(self)
        #for obj in grid.get_agents(pt):
            #if obj.uid[1] == CellulaEpiteliale.TYPE:
                #mucina += 1  #produzione mucina
                #break
    


class LPS(core.Agent):

    TYPE = 1

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = LPS.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def stepMicrobiota(self):
        grid = model.microibiotaGrid
        pt = grid.get_location(self)

        space_pt = model.microbiotaSpace.get_location(self)
        direction = pt.coordinates * 0.4
        model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
  
                 
    def stepLume(self):
        grid = model.lumeGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderLume.find(pt.x, pt.y)

        minimum = [[], sys.maxsize]    
        at = dpt(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == TNFalfa.TYPE:
                    count += 1
            if count < minimum[1]:    
                minimum[0] = [ngh]
                minimum[1] = count
            elif count == minimum[1]:
                minimum[0].append(ngh)

        min_ngh = minimum[0][random.default_rng.integers(0, len(minimum[0]))]

        if not np.all(min_ngh == pt.coordinates):
            space_pt = model.lumeSpace.get_location(self)
            direction = (min_ngh - pt.coordinates) * 0.8
            model.moveLume(self, space_pt.x + direction[0], space_pt.y + direction[1])
        
        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == TNFalfa.TYPE:
                if model.ImmuneResp() == True:
                    model.generate_alfasin(pt)
                break



class CellulaEpiteliale(core.Agent):

    TYPE = 2

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = CellulaEpiteliale.TYPE, rank=rank)
        self.permeability = 10

    def save(self) -> Tuple:
        return (self.uid, self.permeability)
    
    def getPermeability(self):
        return self.permeability

    def step(self):
        if self.permeability <= 80:
            self.permeability += (self.permeability * 5) / 100


class TNFalfa(core.Agent):
    TYPE = 3

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = TNFalfa.TYPE, rank=rank)
        self.rispostaImm=False

    def save(self) -> Tuple:
        return (self.uid, self.rispostaImm)
    
    def getRispostaImm(self):
        return self.rispostaImm
    
    def step(self):
        self.rispostaImm = True

        grid = model.lumeGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderLume.find(pt.x, pt.y)

        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == LPS.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            space_pt = model.lumeSpace.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.5
            model.moveLume(self, space_pt.x + direction[0], space_pt.y + direction[1])


        
class AlfaSinucleina(core.Agent):
    TYPE = 4

    def __init__(self, a_id, rank):
        super().__init__(id = a_id, type = AlfaSinucleina.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def stepLume(self):
        grid = model.lumeGrid
        pt = grid.get_location(self)

        space_pt = model.lumeSpace.get_location(self)
        direction = pt.coordinates * 0.4
        model.moveLume(self, space_pt.x + direction[0], space_pt.y + direction[1])

    def stepNervous(self):
        grid = model.nervousGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderNervous.find(pt.x, pt.y)

        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Nadh.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            space_pt = model.nervousSpace.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.5
            model.moveNervous(self, space_pt.x + direction[0], space_pt.y + direction[1])


class Nadh(core.Agent):
    
    TYPE = 5

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Nadh.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def generate_electron(self, pt):
        # Generate electron when interacting with NADH
        e = Electron(model.electron_id, model.rank)
        model.electron_id += 1
        model.NervousContext.add(e)
        model.moveNervous(e, pt.x, pt.y)

    def step(self):
        grid = model.nervousGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderNervous.find(pt.x, pt.y)

        minimum = [[], sys.maxsize]    
        at = dpt(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == AlfaSinucleina.TYPE:
                    count += 1
            if count < minimum[1]:    
                minimum[0] = [ngh]
                minimum[1] = count
            elif count == minimum[1]:
                minimum[0].append(ngh)

        min_ngh = minimum[0][random.default_rng.integers(0, len(minimum[0]))]

        if not np.all(min_ngh == pt.coordinates):
            space_pt = model.nervousSpace.get_location(self)
            direction = (min_ngh - pt.coordinates) * 0.8
            model.moveNervous(self, space_pt.x + direction[0], space_pt.y + direction[1])
            
        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == AlfaSinucleina.TYPE:
                # release of electron with a 0.8 index of probability
                probability_of_release = 0.8
                if random.default_rng.uniform(0, 1) <= probability_of_release:
                    self.generate_electron(pt)
                break


class Electron(core.Agent):
    
    TYPE = 6

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Electron.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        grid = model.nervousGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderNervous.find(pt.x, pt.y)
        
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at = dpt(*ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Oxygen.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            space_pt = model.nervousSpace.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.7
            model.moveNervous(self, space_pt.x + direction[0], space_pt.y + direction[1])


class Oxygen(core.Agent):
    
    TYPE = 7

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Oxygen.TYPE, rank=rank)
        self.ElectronFusion = False

    def save(self) -> Tuple:
        return (self.uid, self.ElectronFusion)

    def generate_ros(self, pt):
        # Generate electron when interacting with NADH
        r = ROS(model.ros_id, model.rank)
        model.ros_id += 1
        model.NervousContext.add(r)
        model.moveNervous(r, pt.x, pt.y)

    def step(self):
        grid = model.nervousGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderNervous.find(pt.x, pt.y)
        
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at = dpt(*ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Electron.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            space_pt = model.nervousSpace.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.7
            model.moveNervous(self, space_pt.x + direction[0], space_pt.y + direction[1])
            
        
        pt = grid.get_location(self)        
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Electron.TYPE:
                # Reaction with electron to produce ROS
                # Remove the electron and create ROS
                self.ElectronFusion = True
                self.generate_ros(pt)
                model.NervousContext.remove(obj)
                break
        
        return(self.ElectronFusion)
    

class ROS(core.Agent):

    TYPE = 8

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=ROS.TYPE, rank=rank)
    
    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        grid = model.nervousGrid
        pt = grid.get_location(self)
        nghs = model.ngh_finderNervous.find(pt.x, pt.y)

        minimum = [[], sys.maxsize]    
        at = dpt(0, 0)
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Nadh.TYPE:
                    count += 1
            if count < minimum[1]:    
                minimum[0] = [ngh]
                minimum[1] = count
            elif count == minimum[1]:
                minimum[0].append(ngh)

        min_ngh = minimum[0][random.default_rng.integers(0, len(minimum[0]))]

        if not np.all(min_ngh == pt.coordinates):
            space_pt = model.nervousSpace.get_location(self)
            direction = (min_ngh - pt.coordinates) * 0.3
            model.moveNervous(self, space_pt.x + direction[0], space_pt.y + direction[1]) 





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
    
    if uid[1] == TNFalfa.TYPE:
        if uid in agent_cache:
            t = agent_cache[uid]
        else:
            t = TNFalfa(uid[0], uid[2])
            agent_cache[uid] = t

        t.rispostaImm = agent_data[1]
        return t   

    if uid[1] == AlfaSinucleina.TYPE:                                                       
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            a = AlfaSinucleina(uid[0], uid[2])
            agent_cache[uid] = a
            return a



@dataclass
class MicrobiotaCounts:
    scfa: int = 0
    lps: int = 0
    permeability: float = 0.0
    cellEpit: int = 0

@dataclass
class LumeCounts:
    lps: int = 0
    tnfAlfa: int = 0
    #immRespo: bool = False
    alfasin: int = 0

@dataclass
class NervousCounts:
    nadh: int = 0
    alfasinucleina: int = 0
    ros: int = 0
    artificialAgent: int = 0
    electron: int = 0
    oxygen: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.MicrobiotaContext = ctx.SharedContext(comm)
        self.LumeContext = ctx.SharedContext(comm)
        self.NervousContext = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_endMicrobiota)
        self.runner.schedule_end_event(self.at_endLume)
        self.runner.schedule_end_event(self.at_endNervous)

        #Microbiota world
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

        #Lume world
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

        #Nervous world
        box3 = space.BoundingBox(0, params['nervousWorld.width'], 0, params['nervousWorld.height'], 0, 0)    
        self.nervousGrid = space.SharedGrid('grid', bounds=box3, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)    
        self.NervousContext.add_projection(self.nervousGrid)

        self.nervousSpace = space.SharedCSpace('space', bounds=box3, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)    
        self.NervousContext.add_projection(self.nervousSpace)

        self.ngh_finderMicrobiota = GridNghFinder(0, 0, box1.xextent, box1.yextent)
        self.ngh_finderLume = GridNghFinder(0, 0, box2.xextent, box2.yextent)
        self.ngh_finderNervous = GridNghFinder(0, 0, box3.xextent, box3.yextent)

        #logging
        self.microbiotaCounts = MicrobiotaCounts()    
        microbiotaLoggers = logging.create_loggers(self.microbiotaCounts, op=MPI.SUM, rank=self.rank)    
        self.microbiotaData_set = logging.ReducingDataSet(microbiotaLoggers, self.comm, params['logging_file']) 

        self.lumeCounts = LumeCounts()    
        lumeLoggers = logging.create_loggers(self.lumeCounts, op=MPI.SUM, rank=self.rank)    
        self.lumeData_set = logging.ReducingDataSet(lumeLoggers, self.comm, params['lumeLogging_file']) 

        self.nervousCounts = NervousCounts()    
        nervousLoggers = logging.create_loggers(self.nervousCounts, op=MPI.SUM, rank=self.rank)    
        self.nervousData_set = logging.ReducingDataSet(nervousLoggers, self.comm, params['nervousLogging_file'])

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
        
        self.min_scfa = None  #variabile per calcolare il numero minimo di scfa per l'aumento della permeabilità


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
        pp_lpsLume_count = int(total_lpsLume_count / world_size)   
        if self.rank < total_lpsLume_count % world_size:    
            pp_lpsLume_count += 1

        local_boundsLume = self.lumeSpace.get_local_bounds()    
        for i in range(pp_lpsLume_count):    
            p = LPS(i, self.rank)    
            self.LumeContext.add(p)    
            x = random.default_rng.uniform(local_boundsLume.xmin, local_boundsLume.xmin + local_boundsLume.xextent)    
            y = random.default_rng.uniform(local_boundsLume.ymin, local_boundsLume.ymin + local_boundsLume.yextent)
            self.moveLume(p, x, y)
        
        self.lps_id = pp_lpsLume_count
        
        #add tnf to lume context
        total_tnf_count = params['tnf.count']    
        pp_tnf_count = int(total_tnf_count / world_size)  
        if self.rank < total_tnf_count % world_size:    
            pp_tnf_count += 1

        local_boundsLume = self.lumeSpace.get_local_bounds()    
        for i in range(pp_tnf_count):    
            t = TNFalfa(i, self.rank)    
            self.LumeContext.add(t)    
            x = random.default_rng.uniform(local_boundsLume.xmin, local_boundsLume.xmin + local_boundsLume.xextent)    
            y = random.default_rng.uniform(local_boundsLume.ymin, local_boundsLume.ymin + local_boundsLume.yextent)
            self.moveLume(t, x, y)


         #add alfasinucleina to lume context
        total_alfa_count = params['alfasinucleina.count']    
        pp_alfa_count = int(total_alfa_count / world_size)  
        if self.rank < total_alfa_count % world_size:    
            pp_alfa_count += 1

        local_boundsLume = self.lumeSpace.get_local_bounds()    
        for i in range(pp_alfa_count):    
            a = AlfaSinucleina(i, self.rank)    
            self.LumeContext.add(a)    
            x = random.default_rng.uniform(local_boundsLume.xmin, local_boundsLume.xmin + local_boundsLume.xextent)    
            y = random.default_rng.uniform(local_boundsLume.ymin, local_boundsLume.ymin + local_boundsLume.yextent)
            self.moveLume(a, x, y)

        self.alfa_id = pp_alfa_count

        
    def at_endMicrobiota(self):
        self.microbiotaData_set.close()
    
    def at_endLume(self):
        self.lumeData_set.close()

    def at_endNervous(self):
        self.nervousData_set.close()

    def move(self, agent, x, y):
        self.microbiotaSpace.move(agent, cpt(x, y))
        self.microibiotaGrid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))
    
    def moveLume(self, agent, x, y):
        self.lumeSpace.move(agent, cpt(x, y))
        self.lumeGrid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def moveNervous(self, agent, x, y):
        self.nervousSpace.move(agent, cpt(x, y))
        self.nervousGrid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))


    def step(self):
        tick = self.runner.schedule.tick    
        self.log_countsMicrobiota(tick) 
        self.MicrobiotaContext.synchronize(restore_agent)

        self.LumeContext.synchronize(restore_agent)
        self.log_countsLume(tick) 

        scfa_count = []
        for s in self.MicrobiotaContext.agents(SCFA.TYPE):
            s.step()
            scfa_count.append(s)
        
        lps_moved = []
        max_lps_moved = 10 

        for l1 in self.MicrobiotaContext.agents(LPS.TYPE):
            l1.stepMicrobiota()  
            if self.permeability() >= 60:
                if len(lps_moved) < max_lps_moved:
                    lps_moved.append(l1)
        
        for i in lps_moved:
            self.MicrobiotaContext.remove(i)
            self.moveToLume()
        
        for l2 in self.LumeContext.agents(LPS.TYPE):
            l2.stepLume()

        for t in self.LumeContext.agents(TNFalfa.TYPE):
            if self.getNumberLPS() >= 100:
                t.step()

        if tick == 1:
            num_tot_scfa = len(scfa_count)
            self.min_scfa = num_tot_scfa - (num_tot_scfa * 20) / 100  

        for c in self.MicrobiotaContext.agents(CellulaEpiteliale.TYPE):
            if len(scfa_count) <= self.min_scfa:
                c.step()

        for a in self.LumeContext.agents(AlfaSinucleina.TYPE):
            a.stepLume()
        
        if tick >= 10:
            self.removeSCFA() 
        

    def getNumberLPS(self):
        lps_lume = []
        for i in self.LumeContext.agents(LPS.TYPE):
            lps_lume.append(i)

        return len(lps_lume)

    def moveToLume(self):
        local_boundsLume = self.lumeSpace.get_local_bounds()
        l = LPS(self.lps_id, self.rank)
        self.lps_id += 1
        self.LumeContext.add(l)
        x = random.default_rng.uniform(local_boundsLume.xmin, local_boundsLume.xmin + local_boundsLume.xextent)    
        y = random.default_rng.uniform(local_boundsLume.ymin, local_boundsLume.ymin + local_boundsLume.yextent)
        self.moveLume(l, x, y)

    def generate_alfasin(self, pt):
            a = AlfaSinucleina(self.alfa_id, self.rank)
            self.alfa_id += 1
            self.LumeContext.add(a)
            self.moveLume(a, pt.x, pt.y) 

    def ImmuneResp(self):
        for i in self.LumeContext.agents(TNFalfa.TYPE):
           immuneResponse = i.getRispostaImm()

        return immuneResponse

    def permeability(self):
        for ce in self.MicrobiotaContext.agents(CellulaEpiteliale.TYPE):
            permeability = ce.getPermeability()

        return permeability

    def removeSCFA(self):
        num_SCFA = []
        for i in self.MicrobiotaContext.agents(SCFA.TYPE):
            num_SCFA.append(i)
            
        countSCFA = len(num_SCFA)
        if countSCFA > 10: 
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
        num_LumeAgents= self.LumeContext.size([LPS.TYPE, TNFalfa.TYPE, AlfaSinucleina.TYPE])

        self.lumeCounts.lps = num_LumeAgents[LPS.TYPE]
        self.lumeCounts.tnfAlfa = num_LumeAgents[TNFalfa.TYPE]
        #self.lumeCounts.immRespo = self.ImmuneResp()
        self.lumeCounts.alfasin = num_LumeAgents[AlfaSinucleina.TYPE]

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