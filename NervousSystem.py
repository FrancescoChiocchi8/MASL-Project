import math
import sys
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


class AlfaSinucleina(core.Agent):

    TYPE = 0

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=AlfaSinucleina.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)

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
            space_pt = model.space.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.5
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])

        


class Nadh(core.Agent):
    
    TYPE = 1

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Nadh.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)
    
    def generate_electron(self, pt):
        # Generate electron when interacting with NADH
        e = Electron(model.electron_id, model.rank)
        model.electron_id += 1
        model.context.add(e)
        model.move(e, pt.x, pt.y)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)

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
            space_pt = model.space.get_location(self)
            direction = (min_ngh - pt.coordinates) * 0.8
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
            
        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == AlfaSinucleina.TYPE:
                # release of electron with a 0.8 index of probability
                probability_of_release = 0.8
                if random.default_rng.uniform(0, 1) <= probability_of_release:
                    self.generate_electron(pt)
                break
            
class ROS(core.Agent):

    TYPE = 2

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=ROS.TYPE, rank=rank)
    
    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)

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
            space_pt = model.space.get_location(self)
            direction = (min_ngh - pt.coordinates) * 0.3
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])  


class ArtificialAgent(core.Agent):

    TYPE = 3

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=ArtificialAgent.TYPE, rank=rank)
    
    def save(self) -> Tuple:
        return (self.uid,)
    
    def remove_agentROS(self, agent):
        model.context.remove(agent)
    
    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)
        cpt = model.space.get_location(self)

        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(ngh)
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == ROS.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)
        
        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not np.all(max_ngh == pt.coordinates):
            direction = (max_ngh - pt.coordinates[0:3]) * 0.4
            model.move(self, cpt.x + direction[0], cpt.y + direction[1])

        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == ROS.TYPE:
                self.remove_agentROS(obj)
                break
        

class Electron(core.Agent):
    
    TYPE = 4

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Electron.TYPE, rank=rank)

    def save(self) -> Tuple:
        return (self.uid,)

    def step(self):
        # Move the electron
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)
        
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
            space_pt = model.space.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.7
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])


class Oxygen(core.Agent):
    
    TYPE = 5

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Oxygen.TYPE, rank=rank)
        self.ElectronFusion = False

    def save(self) -> Tuple:
        return (self.uid, self.ElectronFusion)

    def generate_ros(self, pt):
        # Generate electron when interacting with NADH
        r = ROS(model.ros_id, model.rank)
        model.ros_id += 1
        model.context.add(r)
        model.move(r, pt.x, pt.y)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        nghs = model.ngh_finder.find(pt.x, pt.y)
        
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
            space_pt = model.space.get_location(self)
            direction = (max_ngh - pt.coordinates[0:3]) * 0.7
            model.move(self, space_pt.x + direction[0], space_pt.y + direction[1])
            
        
        pt = grid.get_location(self)        
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Electron.TYPE:
                # Reaction with electron to produce ROS
                # Remove the electron and create ROS
                self.ElectronFusion = True
                self.generate_ros(pt)
                model.context.remove(obj)
                #model.context.remove(self)
                break
        
        return(self.ElectronFusion)


agent_cache = {} 

def restore_agent(agent_data: Tuple): 
    #uid element 0 is id, 1 is type, 2 is rank
    uid = agent_data[0]                                         
    
    if uid[1] == Nadh.TYPE:                                      
        if uid in agent_cache:                                  
            return agent_cache[uid]
        else:
            n = Nadh(uid[0], uid[2])
            agent_cache[uid] = n
            return n
    
    if uid[1] == Electron.TYPE:                                      
        if uid in agent_cache:                                  
            return agent_cache[uid]
        else:
            e = Electron(uid[0], uid[2])
            agent_cache[uid] = e
            return e
        
    if uid[1] == Oxygen.TYPE:                                      
        if uid in agent_cache:                                  
            o = agent_cache[uid]
        else:
            o = Oxygen(uid[0], uid[2])
            agent_cache[uid] = o
        
        o.ElectronFusion = agent_data[1]
        return o
        
    if uid[1] == AlfaSinucleina.TYPE:                                                       
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            a = AlfaSinucleina(uid[0], uid[2])
            agent_cache[uid] = a
            return a
        
    if uid[1] == ROS.TYPE:                                                       
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            r = ROS(uid[0], uid[2])
            agent_cache[uid] = r
            return r
        
    if uid[1] == ArtificialAgent.TYPE:                                                       
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            q = ArtificialAgent(uid[0], uid[2])
            agent_cache[uid] = q
            return q


@dataclass
class Counts:
    nadh: int = 0
    alfasinucleina: int = 0
    ros: int = 0
    artificialAgent: int = 0
    electron: int = 0
    oxygen: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        self.electron_id = 0

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)    
        self.grid = space.SharedGrid('grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)    
        self.context.add_projection(self.grid)    
        self.space = space.SharedCSpace('space', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)    
        self.context.add_projection(self.space)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)


        #logging
        self.counts = Counts()    
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)    
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['logging_file']) 

        world_size = comm.Get_size()

        #add nadh agents to context
        total_nadh_count = params['nadh.count']    
        pp_nadh_count = int(total_nadh_count / world_size)   #number of nadh per processor 
        if self.rank < total_nadh_count % world_size:    
            pp_nadh_count += 1
        
        local_bounds = self.space.get_local_bounds()    
        for i in range(pp_nadh_count):    
            h = Nadh(i, self.rank)    
            self.context.add(h)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(h, x, y) 
        

        #add Alfasinucleina agents to context
        total_alfa_count = params['alfasinucleina.count']
        pp_alfa_count = int(total_alfa_count / world_size)
        if self.rank < total_alfa_count % world_size:
            pp_alfa_count += 1
        
        for i in range(pp_alfa_count):
            alf = AlfaSinucleina(i, self.rank)
            self.context.add(alf)
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(alf, x, y)
        

        #add ros agents to context
        total_ros_count = params['ros.count']    
        pp_ros_count = int(total_ros_count / world_size)   #number of ros per processor 
        if self.rank < total_ros_count % world_size:    
            pp_ros_count += 1
        
        local_bounds = self.space.get_local_bounds()    
        for i in range(pp_ros_count):    
            r = ROS(i, self.rank)    
            self.context.add(r)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(r, x, y) 

        self.ros_id = pp_ros_count
        
        #add artificialAgent agents to context
        total_aa_count = params['artificialagent.count']    
        pp_aa_count = int(total_aa_count / world_size) 
        if self.rank < total_aa_count % world_size:    
            pp_aa_count += 1
        
        local_bounds = self.space.get_local_bounds()  
        for i in range(pp_aa_count):    
            q = ArtificialAgent(i, self.rank)    
            self.context.add(q)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(q, x, y) 
            
        #add electron agents to context
        total_el_count = params['electron.count']    
        pp_el_count = int(total_el_count / world_size) 
        if self.rank < total_el_count % world_size:    
            pp_el_count += 1
        
        local_bounds = self.space.get_local_bounds()  
        for i in range(pp_el_count):    
            e = Electron(i, self.rank)    
            self.context.add(e)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(e, x, y) 
        
        self.electron_id = pp_el_count
            
        #add oxygen agents to context
        total_ox_count = params['oxygen.count']    
        pp_ox_count = int(total_ox_count / world_size) 
        if self.rank < total_ox_count % world_size:    
            pp_ox_count += 1
        
        local_bounds = self.space.get_local_bounds()  
        for i in range(pp_ox_count):    
            o = Oxygen(i, self.rank)    
            self.context.add(o)    
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)    
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            self.move(o, x, y) 


    def at_end(self):
        self.data_set.close()
        

    def move(self, agent, x, y):
        self.space.move(agent, cpt(x, y))
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))
    

    def step(self):
        tick = self.runner.schedule.tick    
        self.log_counts(tick)    
        self.context.synchronize(restore_agent)    

        for z in self.context.agents(AlfaSinucleina.TYPE):
            z.step()
   
        for e in self.context.agents(Electron.TYPE):    
            e.step()
        
        oxigen_fusion = []
        for o in self.context.agents(Oxygen.TYPE):
            fusion = o.step()
            if fusion == True:
                oxigen_fusion.append(o)
                
        for i in oxigen_fusion:
            self.context.remove(i)
            
        for h in self.context.agents(Nadh.TYPE):    
            h.step()
        
        for r in self.context.agents(ROS.TYPE):
            r.step()

        for q in self.context.agents(ArtificialAgent.TYPE):
            q.step()
            

    def run(self):
        self.runner.execute()
        
    
    def log_counts(self, tick):
        num_agents = self.context.size([Nadh.TYPE, AlfaSinucleina.TYPE, ROS.TYPE, ArtificialAgent.TYPE, Electron.TYPE, Oxygen.TYPE])    
        self.counts.nadh = num_agents[Nadh.TYPE]    
        self.counts.alfasinucleina = num_agents[AlfaSinucleina.TYPE] 
        self.counts.ros = num_agents[ROS.TYPE]      
        self.counts.artificialAgent = num_agents[ArtificialAgent.TYPE] 
        self.counts.electron = num_agents[Electron.TYPE]
        self.counts.oxygen = num_agents[Oxygen.TYPE]
        self.data_set.log(tick)


def run(params: Dict):
    global model    
    model = Model(MPI.COMM_WORLD, params)   
    model.run()


if __name__ == "__main__":
    parser = parameters.create_args_parser()    
    args = parser.parse_args()   
    params = parameters.init_params(args.parameters_file, args.parameters)    
    run(params)