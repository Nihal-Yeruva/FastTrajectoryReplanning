import random

class Maze:

    #initialization
    def __init__(self, size):
        self.size = size
        self.grid = [['U' for _ in range(size)] for _ in range(size)]
        self.visited = [[False for _ in range(size)] for _ in range(size)]
        self.stack = []
    
    #                                                                  down   up   right  left
    #get all unvisited neighbors of a cell (adjacent not diagonals)  [(1,0),(-1,0),(0,1),(0,-1)]
    def getNeighbors(self, x, y):
        neighbours = []
        if x > 0 and not self.visited[x - 1][y]:
            neighbours.append((x - 1, y))
        if x < self.size - 1 and not self.visited[x + 1][y]:
            neighbours.append((x + 1, y))
        if y > 0 and not self.visited[x][y - 1]:
            neighbours.append((x, y - 1))
        if y < self.size - 1 and not self.visited[x][y + 1]:
            neighbours.append((x, y + 1))
        return neighbours

    #check if all the cells have been visited
    def allVisited(self):
        for i in range(self.size):
            for j in range(self.size):
                if not self.visited[i][j]:
                    return False
        return True

    #maze generation
    def generate(self):
        #start from a random cell
        startCellx = random.randint(0, self.size - 1)
        startCelly = random.randint(0, self.size - 1)
        self.stack.append((startCellx, startCelly))
        self.visited[startCellx][startCelly] = True
        #unblocked cells will be marked 0, while blocked cells will be marked 1
        self.grid[startCellx][startCelly] = '0'

        #random tie-breaking generating unblocked and blocked cells (0 for unblocked, 1 for blocked)
        while self.stack:

            x, y = self.stack[-1]
            neighbours = self.getNeighbors(x, y)

            if neighbours:
                nextCellx, nextCelly = random.choice(neighbours)
                if random.random() < 0.7:
                    self.stack.append((nextCellx, nextCelly))
                    self.grid[nextCellx][nextCelly] = '0'
                else:
                    self.grid[nextCellx][nextCelly] = '1'

                self.visited[nextCellx][nextCelly] = True
            else:
                self.stack.pop()

            if not self.stack and not self.allVisited():
                for i in range(self.size):
                    for j in range(self.size):
                        if not self.visited[i][j]:
                            self.stack.append((i, j))
                            self.visited[i][j] = True
                            self.grid[i][j] = '0'
                            break
                    if self.stack:
                        break

    
#store mazes as txt files
def saveMazes(maze, index):
    with open(f'maze_{index}.txt', 'w') as f:
        for row in maze.grid:
            f.write(' '.join(row) + '\n')

#maze generation and storing (as txt files)
def main():
    size = 101
    numOfgrids = 50

    for i in range(numOfgrids):
        maze = Maze(size)
        maze.generate()
        saveMazes(maze, i)

# if __name__ == "__main__":
#     main()
