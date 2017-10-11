import matplotlib.pyplot as plt

class Actor:
    def __init__(self, pos, init_state=None):
        self.pos = pos
        self.state = init_state
        self.world = None

    def act(self):
        # It is up to the user to define how an actor will act.
        pass

    def color(self):
        return '#000000'
    
    def draw(self, plt):
        # Draw the actor
        plt.scatter(
                self.pos[0],
                self.pos[1], 
                color=self.color()
        )
     


class Character(Actor):
    def __init__(self, pos, gene = None, init_state = None):
        # Call to supertype
        super(Character, self).__init__(pos, init_state)
        
        # Store the DNA sequence.
        self.dna = gene

    def random_gene(self):
        # It is up to the user to define DNA sequences.
        return None

    def color(self):
        return '#00ff00'
    

class World:

    def __init__(self):
        self.actors = []

    def draw_world(self):
        actors = self.actors

        # Perform draw operation
        plt.clf()

        for a in actors:
            # Get color and position.
            color = a.color()
            pos = a.pos
            
            a.draw(plt)
        
        # Finalize plot
        plt.xlim(-32, 32)
        plt.ylim(-32, 32)
        plt.pause(0.01)

    def run_cycle(self, redraw=True):
        # Have all actors act.
        for a in self.actors:
            a.act()
        
        if redraw:
            self.draw_world()
    
    def add_actor(self, actor):
        self.actors.append(actor)

        # The actor now belongs to this world.
        actor.world = self

    def rem_actor(self, actor):
        self.actors.remove(actor)

        # The actor no longer belongs to a world.
        actor.world = None



