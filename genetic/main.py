# My code
from world import *

# Third-party APIs
import numpy as np

# Standard libs
import math, random

# Time shift per frame
dt = 0.125

class Food(Actor):
    def __init__(self, pos, amt = None):
        if amt == None:
            # Choose a random amount of food value
            amt = math.sqrt(np.random.normal(0, 0.25)**2)

        super(Food, self).__init__(pos, amt)


class EvoWorld(World):
    
    def __init__(self):
        super(EvoWorld, self).__init__()
        self.graveyard = []

    
class Critter(Character):

    def __init__(self, pos, gene=None):
        # Provide a DNA sequence if necessary
        g = gene
        if gene == None:
            g = Critter.random_gene()
    
        # State is {vel, facing, awareness, health, reccurent_state}
        state = (0, random.random()*2*math.pi, 0.5, 1.0, 0)

        super(Critter, self).__init__(pos, gene = g, init_state = state)

        self.age = 0

    def random_gene():
        """ Creates a genetic code in the form of a neural network
        """
        # Creatures get their health state and what they can see
        # inputs: < health, food_seen, critters_seen >

        # Using this, they compute what to do (how much awareness, how to move)
        # outputs: < forward_speed, rotation, awareness >

        genome = []

        # Provide a random mutation parameter (a mutator)
        genome.append(math.sqrt(np.random.normal(0, 1)**2))

        # Give a two-layer network.
        genome.append(np.random.normal(0, 1, size=(5, 4)))
        genome.append(np.random.normal(0, 1, size=(4, 5)))

        return genome

    def mutate_gene(self, gene=None):
        if gene == None:
            gene = self.dna

        genome = []
        
        # Mutate the mutator
        genome.append(abs(gene[0] + math.sqrt(np.random.normal(0, 1)**2)))
        
        # Mutate the network
        genome.append(gene[1] + np.random.normal(0, gene[0], size=(5,4)))
        genome.append(gene[2] + np.random.normal(0, gene[0], size=(4,5)))

        return genome

    def net_output(self):
        world = self.world
        state = self.state

        rot = state[1]
        aware = state[2]

        seen_critters = 0
        seen_food = 0
        for a in world.actors:
            d = np.dot(np.array([math.cos(rot), math.sin(rot)]), a.pos - self.pos)
            if d < aware * np.linalg.norm(a.pos-self.pos):
                continue
            elif type(a) is Critter:
                seen_critters += 1
            elif type(a) is Food:
                seen_food += 1

        x = np.array([
                # Current health
                self.state[3],
                # Number of visible food units
                seen_food,
                # Number of critters seen
                seen_critters,
                # Recurrent state parameter
                state[3]
        ])

        # Propagate across all layers
        Wx = np.matmul(self.dna[1], x)
        y = np.vectorize(lambda v : 0 if v < -512 else 1.0 / (1.0 + math.exp(-v)))(Wx)
        y = np.matmul(self.dna[2], y)
        
        return y

    
    def act(self):
        # Compute an action vector
        action = self.net_output()

        # Velocity bounded between 0 and 1
        vel = 5.0 * (1.5 * (1.0 / (1.0 + math.exp(-action[0])) * dt) - 0.5)
        
        # Allow any rotation change
        rot = self.state[1] + action[1]
        while rot >= 2*math.pi:
            rot -= 2*math.pi
        while rot < 0:
            rot += 2*math.pi
        
        # Recalculate awareness (cosine of angle btween forward and vision border)
        awareness = 2.0*(1.0 / (1.0 + math.exp(-action[2]))) - 1.0

        # Move the creature
        self.pos = np.clip(self.pos + np.array([vel*math.cos(rot), vel*math.sin(rot)]), -32, 32)

        health = self.state[3]

        for a in self.world.actors:
            if type(a) is Food and np.linalg.norm(a.pos - self.pos) < 1.0:
                # Feed on the food if within 0.5 units of the food
                self.world.rem_actor(a)
                health += a.state

                # Replenish the food supply
                #self.world.add_actor(Food(np.random.uniform(-32, 32, size=(2,))))

        # Health gradually deteriorates (faster when moving)
        health -= (0.01*abs(vel) + 0.01) * dt

        if health <= 0:
            # If dead, remove oneself from the world
            self.world.graveyard.append(self)
            self.world.rem_actor(self)
        elif health > 2:
            # If health is very good, make a baby
            health -= 1
            self.world.add_actor(Critter(self.pos, self.mutate_gene()))

        
        # Update the state
        self.state = (vel, rot, awareness, health)
        
        # Increase age
        self.age += dt

    def color(self):
        health = max(0, min(self.state[3], 1))
        return ('#' +
                "{:02x}".format(255 - int(255*health)) +
                "{:02x}".format(int(255*health)) +
                '00')

    def draw(self, plt):
        
        super(Critter, self).draw(plt)

        angle = self.state[1]
        aw_ang = math.acos(self.state[2])
        
        plt.plot(
            [self.pos[0], self.pos[0] + 2*math.cos(angle + aw_ang)],
            [self.pos[1], self.pos[1] + 2*math.sin(angle + aw_ang)],
            'k-')

        plt.plot(
            [self.pos[0], self.pos[0] + math.cos(angle)],
            [self.pos[1], self.pos[1] + math.sin(angle)],
            'k-')

        plt.plot(
            [self.pos[0], self.pos[0] + 2*math.cos(angle - aw_ang)],
            [self.pos[1], self.pos[1] + 2*math.sin(angle - aw_ang)],
            'k-')

n_surv = 4
n_child = 2
f_supply = 20

p_size = n_surv * (1 + n_child) + 3

world = EvoWorld()

for _ in range(p_size): 
    world.add_actor(Critter(np.random.uniform(-32, 32, size=(2,))))

for _ in range(f_supply):
    world.add_actor(Food(np.random.uniform(-32, 32, size=(2,))))

epoch=0
cycle = 0

print('Begin')
p = 0

while True:
    if any([type(a) is Critter for a in world.actors]):
        # There are still living beings
        world.run_cycle(epoch >= 0 and cycle%10 == 0)
        cycle += 1
        tmp = len([a for a in world.actors if type(a) is Critter])
        if tmp != p:
            if len(world.graveyard) > n_surv:
                # Cremate the inferior dead
                world.graveyard.remove(min(world.graveyard, key=lambda x : x.age))

            p = tmp
            print('popsize:', p)

        if len(world.actors) - p < f_supply and cycle % int(len(world.actors)):
            world.add_actor(Food(np.random.uniform(-32, 32, size=(2,))))

    else:
        print('End round', epoch+1)

        # Reset the world
        graves = world.graveyard
        
        # Get the best cases
        surv = list(reversed(sorted(world.graveyard, key=lambda x : x.age)))[:n_surv]
        print('highest age:', surv[0].age)

        # Add initial DNA samples
        genes = [s.dna for s in surv]

        # Make batches of children
        for i in range(n_child):
            genes += [s.mutate_gene() for s in surv]
        
        # Add random organisms to fill in the gap
        genes += [Critter.random_gene() for _ in range(p_size - n_surv*(1+n_child))]
        
        # Make a new world
        world = EvoWorld()
        
        # Add new organisms
        for g in genes:
            world.add_actor(Critter(np.random.uniform(-32, 32, size=(2,)), g))
        
        # Add food
        for _ in range(f_supply):
            world.add_actor(Food(np.random.uniform(-32, 32, size=(2,))))

        epoch += 1
        cycle = 0
            
print('Game over')

