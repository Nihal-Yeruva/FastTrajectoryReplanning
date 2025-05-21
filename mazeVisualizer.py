import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def readTxt(file):
    with open(file, 'r') as f:
        maze = []
        for line in f:
            maze.append(list(map(int, line.split())))
    return maze

def plotMaze(maze):
    mazeArr = np.array(maze)
    plt.imshow(mazeArr, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()

#can be implemented iteratively for all mazes, just implemented for individual visualization for now

#file = 'maze_49.txt'
#maze = readTxt(file)

#plotMaze(maze)