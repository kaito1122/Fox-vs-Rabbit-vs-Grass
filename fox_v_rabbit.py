"""
Jeffrey Pan, Kathryn Warren, and Kaito Minami
DS3500
HW5: Fox vs. Rabbit
created: 4.10.2023
last updated: 4.14.2023
"""

# import statements
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
import argparse

# creating a custom color map
colors = ["tan", "darkgreen", "darkblue", "firebrick"]
cmap = ListedColormap(colors)

parser = argparse.ArgumentParser(
    prog='fox_v_rabbit.py',
    description='Simulates artificial ecosystem with foxes, rabbits, and grass',
    epilog='Text at the bottom of help')

# initializing argparse arguments to edit simulation from command line
# example command line:
# python fox_v_rabbit.py -s=250 -g=0.25 -r=40000 -f=10000 -k=20 -t=1000 -i=1 -fo=3 -ro=4
parser.add_argument('-s','--size',
                    help='the dimension of the field',
                    type=int, default=100)
parser.add_argument('-g','--grass_rate',
                    help='probability that grass grows back at any location in next season', type=float, default = 0.2)
parser.add_argument('-r','--r_init_pop',
                    help='initial number of rabbits in population',
                    type=int, default=100)
parser.add_argument('-f','--f_init_pop',
                    help='initial number of foxes in population',
                    type=int, default=100)
parser.add_argument('-k','--k_cycles',
                    help='number of cycles before the fox dies of starvation',
                    type=int, default=10)
parser.add_argument('-t','--total_gens',
                    help='number of generations to iterate through until program is complete',
                    type=int, default=1000)
parser.add_argument('-i','--interval_speed',
                    help='interval between frames in the animation',
                    type=int, default=1)
parser.add_argument('-fo','--fox_offspring',
                    help='maximum number of offspring foxes can have at a time',
                    type=int, default=1)
parser.add_argument('-ro','--rabbit_offspring',
                    help='maximum number of offspring rabbits can have at a time',
                    type=int, default=2)

# initialize list of argparse arguments
args = parser.parse_args()

# current list of variables that might be nice to pull out and allow the user to edit from command line
SIZE = args.size  # The dimensions of the field
R_OFFSPRING = args.rabbit_offspring  # Max offspring when a rabbit reproduces
F_OFFSPRING = args.fox_offspring  # Max offspring when a fox reproduces
GRASS_RATE = args.grass_rate  # Probability that grass grows back at any location in the next season.
WRAP = True  # Does the field wrap around on itself when rabbits move?
R_INIT_POP = args.r_init_pop  # initial rabbit population
F_INIT_POP = args.f_init_pop  # initial fox population
GRASS_SURVIVE = 0  # rabbits must eat more than this threshold of grass to live another day
RABBIT_SURVIVE = 0  # foxes must eat more than this threshold of rabbits to live another day
TOTAL_GENS = args.total_gens  # number of generations to run the simulation for
K = args.k_cycles # number of cycles before the fox dies of starvation
SPEED = args.interval_speed # interval between frames in the animation

# initializing the 3 classes
class Fox:
    """ Another furry creature roaming the field looking for bunnies to eat. """

    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.cycle = 0

    def reproduce(self):
        """ Make a new fox at the same location.

        args:
            None

        returns:
            copy of the fox
        """
        self.eaten = 0
        # copy will keep the current location already, so convenient!
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the fox a rabbit when they find one.

        args:
            amount(int): amount that was eaten

        returns:
            None
        """
        self.eaten += amount
        self.cycle = 0

    def move(self):
        """ Move up, down, left, right randomly. The fox can move up to 2 spaces at a time

        args:
            None

        returns:
            None
        """
        # WRAP helps address if the fox is at the edge of the array and needs to move
        if WRAP:
            self.x = (self.x + rnd.choice(range(-2,2))) % SIZE
            self.y = (self.y + rnd.choice(range(-2,2))) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice(range(-2,2)))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice(range(-2,2)))))

        self.cycle += 1
class Rabbit:
    """ A furry creature roaming a field in search of grass to eat.
    Mr. Rabbit must eat enough to reproduce, otherwise he starves to death.
    Also the preferred meal of Mr. Fox.
    """
    def __init__(self):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero.

         args:
            None

        returns:
            copy of rabbit
        """
        # set eaten to 0
        self.eaten = 0

        # copy
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the rabbit some grass

        args:
            amount(int): amount that was eaten

        returns:
            None
        """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right randomly. The fox can move up to 2 spaces at a time

        args:
            None

        returns:
            None
        """
        # WRAP helps address if the fox is at the edge of the array and needs to move
        if WRAP:
            self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
            self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-1, 0, 1]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits or foxes """
        self.rabbits = []
        self.foxes = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.nrabbits = []
        self.ngrass = []
        self.nfoxes = []
        self.allfield = np.ones(shape=(SIZE, SIZE), dtype=int)

    def add_fox(self, fox):
        """" A new fox is added to the ecosystem

        args:
            fox(Fox): a Fox object

        returns:
            None
        """
        # add fox
        self.foxes.append(fox)

    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field

        args:
            rabbit(Rabbit): a Rabbit object

        returns:
            None
        """
        # add rabbit
        self.rabbits.append(rabbit)

    def eat_rabbit(self, rabbit):
        """ Fox eats a rabbit. Goodbye rabbit.

        args:
            rabbit(Rabbit): a Rabbit object

        returns:
            None
        """
        # remove rabbit
        self.rabbits.remove(rabbit)

    def move(self):
        """ Rabbits and foxes move based on their capabilities

        args:
            None

        returns:
            None
        """
        # move the animals
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            f.move()

    def eat(self):
        """ Rabbits eat (if they find grass where they are)
         Foxes eat (if they find rabbits where they are)

         args:
            None

        returns:
            None
        """

        # for each rabbit in list, have it eat and then mark grass as gone
        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x, rabbit.y])
            # mark that the grass is gone
            self.field[rabbit.x, rabbit.y] = 0

        # for each fox in list, have it eat if the fox is on the same "cell" as rabbit
        for fox in self.foxes:
            for rabbit in self.rabbits:
                if [rabbit.x, rabbit.y] == [fox.x, fox.y]:
                    self.eat_rabbit(rabbit)
                    # foxes get 1 energy per rabbit
                    fox.eat(1)

    def survive(self):
        """ Rabbits who eat some grass live to eat another day, foxes who eat once every ten cycles survive

        args:
            None

        returns:
            None
        """
        # rabbits' survival is at risk every cycle
        self.rabbits = [r for r in self.rabbits if r.eaten > GRASS_SURVIVE]

        # if it survives K cycles, keep in list, otherwise remove
        self.foxes = [f for f in self.foxes if f.cycle < K]

    def r_reproduce(self):
        """ Rabbits reproduce like rabbits. Rabbits must have at least 1 eaten to reproduce.

        args:
            None

        returns:
            None
        """
        # born list
        born = []

        # for each rabbit, if they have eaten then reproduce
        for rabbit in self.rabbits:
            if rabbit.eaten > 0:
                for _ in range(rnd.randint(1, R_OFFSPRING)):
                    born.append(rabbit.reproduce())
        self.rabbits += born

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.ngrass.append(self.amount_of_grass())

    def f_reproduce(self):
        """ Foxes reproduce like foxes. Foxes must have at least 1 eaten to reproduce.

        args:
            None

        returns:
            None
        """
        # born list
        born = []

        # for each fox, if they have eaten then reproduce
        for fox in self.foxes:
            if fox.eaten > 0:
                for _ in range(rnd.randint(1, F_OFFSPRING)):
                    born.append(fox.reproduce())

        self.foxes += born
        # Capture field state for historical tracking
        self.nfoxes.append(self.num_foxes())

    def grow(self):
        """ Grass grows back with some probability

        args:
            None
        returns:
            None
        """

        # set array
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_animals(self):
        """ Obtain all animals

        args:
            None

        returns:
            None
        """
        self.allfield = np.maximum(self.field, np.maximum(2*self.get_rabbits(), 3*self.get_foxes()))

    def get_rabbits(self):
        """ get the coordinate of all rabbits on field

        args:
            None

        returns:
            rabbits(array): rabbit array
        """
        # initialize rabbits array
        rabbits = np.zeros(shape=(SIZE,SIZE), dtype=int)

        # set coordinates
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1

        return rabbits

    def get_foxes(self):
        """ get the coordinate of all foxes on field

        args:
            None

        returns:
            foxes(array): foxes array
        """
        # initialize foxes array
        foxes = np.zeros(shape=(SIZE,SIZE), dtype=int)

        # set coordinates
        for f in self.foxes:
            foxes[f.x, f.y] = 1
        return foxes

    def num_rabbits(self):
        """ How many rabbits are there in the field?

        args:
            None

        returns:
            length of rabbits list
        """
        return len(self.rabbits)

    def num_foxes(self):
        """ How many foxes are there in the field?

        args:
            None

        returns:
            length of fox list
        """
        return len(self.foxes)

    def amount_of_grass(self):
        """ calculates the amount of grass

        args:
            None

        returns:
            sum of field array
        """
        # return sum
        return self.field.sum()

    def generation(self):
        """ Run one generation of animals

        args:
            None

        returns:
            None
        """

        # run functions
        self.move()
        self.eat()
        self.survive()
        self.r_reproduce()
        self.f_reproduce()
        self.grow()
        self.get_animals()

    def perc_history(self, showTrack=True, showPercentage=True):
        """ history of the animals and grass

        args:
            showTrack(bool): enables tracking if true
            showPercentage(bool): enables percentages to be shown
            marker(str): marker

        returns:
            a lineplot and history.png
        """
        # initialize figure an axes
        fig, ax = plt.subplots(figsize=(7, 6))
        labels = ['# Rabbits', '# Grass', '# Foxes']

        # if percentage is enabled, set to percentages
        xs = self.nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            labels[0] = labels[0].replace('#', '%')

        # if percentage is enabled, set to percentages
        ys = self.ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            labels[1] = labels[1].replace('#', '%')

        # if percentage is enabled, set to percentages
        zs = self.nfoxes[:]
        if showPercentage:
            maxfox = max(zs)
            zs = [z / maxfox for z in zs]
            labels[2] = labels[2].replace('#', '%')

        # if track is enabled, set as lineplot, else scatter
        if showTrack:
            ax.plot(xs)
            ax.plot(ys)
            ax.plot(zs)
            ax.legend()
        else:
            ax.scatter(xs)
            ax.scatter(ys)
            ax.scatter(zs)
            ax.legend()

        # set title and save the figure
        ax.set_title("Rabbits vs. Grass vs Foxes: GROW_RATE =" + str(GRASS_RATE))
        ax.legend(labels=labels)
        plt.savefig("viz/history.png", bbox_inches='tight')
        plt.show()

    def kde_rf(self):
        """ kernel density estimation of rabbits vs foxes

        args:
            None

        returns:
            a kdeplot and history2.png
        """
        # sets x and y variables
        xs = self.nrabbits[:]
        ys = self.nfoxes[:]

        # initiates figure and axes
        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        # scatterplot with histogram and kernel density estimation
        sns.scatterplot(x=xs, y=ys, s=5, color=".15")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
        plt.grid()
        plt.xlim(0, max(xs)*1.2)

        # sets labels and title
        plt.xlabel("# Rabbits")
        plt.ylabel("# Foxes")
        plt.title("Rabbits vs. Foxes")

        # download the result
        plt.savefig("viz/history2.png", bbox_inches='tight')
        plt.show()

    def kde_grf(self):
        """ 3D kernel density estimation of rabbits vs grass vs foxes

        args:
            None
            
        returns:
            a 3d density-weighted scatterplot and history3.png
        """
        # initiates figure and axes
        fig = plt.figure(figsize=(7, 6))
        ax = plt.axes(projection='3d')

        # sets x, y, and z variables
        xs = self.nrabbits[:]
        ys = self.ngrass[:]
        zs = self.nfoxes[:]

        sns.set_style('dark')

        # determines the density of each scatter points
        # source:
        # https://stackoverflow.com/questions/21918529/multivariate-kernel-density-estimation-in-python
        xyz = np.vstack([xs, ys, zs])
        kde = stats.gaussian_kde(dataset=xyz)
        density = kde(xyz)

        # 3d scatterplot with color-ed density
        ax.scatter(xs, ys, zs, c=density)

        # sets labels and title
        ax.set_xlabel("# Rabbits")
        ax.set_ylabel("# Grass")
        ax.set_zlabel('# Foxes')
        ax.set_title("Rabbits vs. Grass vs Foxes")

        # download the result
        plt.savefig("viz/history3.png", bbox_inches='tight')
        plt.show()

def animate(i, field, im):
    """ animates the field

    args:
        i(int): generation number
        field(Field): Field object
        im(image): image of the field

    returns:
        im(image): image ofthe field
    """
    # generation
    field.generation()

    # create image and set title
    im.set_array(field.allfield)
    plt.title("generation = " + str(i))
    return im


def main():
    # Create the ecosystem
    field = Field()

    # add rabbit
    for _ in range(R_INIT_POP):
        field.add_rabbit(Rabbit())

    # add fox
    for _ in range(F_INIT_POP):
        field.add_fox(Fox())

    # initialize array and create image
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=len(colors))
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames = TOTAL_GENS+1, interval=SPEED, repeat=False)
    plt.show()

    # return information
    print('foxes:', field.nfoxes)
    print('rabbits:', field.nrabbits)
    print('grass:', field.ngrass)
    print(len(field.ngrass))

    # plot the population histories
    field.perc_history()
    field.kde_rf()
    field.kde_grf()


if __name__ == '__main__':
    main()
