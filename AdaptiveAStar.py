import random
import matplotlib
matplotlib.use('TkAgg')  # Ensures the right backend for interactive windows
import matplotlib.pyplot as plt
import numpy as np

from BinaryHeap import BinaryHeap

class AdaptiveAStar:
    def __init__(self, mazeFile, tie_break ='small'):
        self.grid = self.loadMaze(mazeFile)  # Load maze from file
        self.size = len(self.grid)  # Size of the grid (Assuming square grid)
        self.start = (0, 0)  # Starting position (top-left corner)
        self.goal = (self.size - 1, self.size - 1)  # Goal position (bottom-right corner)
        self.tie_break = tie_break  # Tie-breaking strategy for A*
        self.knowledge = [['U' for _ in range(self.size)] for _ in range(self.size)]  # Agent's knowledge of the grid
        self.knowledge[self.start[0]][self.start[1]] = '0'  # Start is known to be unblocked
        self.cells_expanded=0
        self.replans=0
        self.h_values = {}  # New attribute for storing the heuristic values
        # Setup for dynamic visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Adaptive A* Maze Solver')
        self.visited_cells = set()  # Keep track of visited cells
        # Add window close event handler
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
    
        # Add flag to track if window was closed
        self.window_closed = False
        

    def _handle_close(self, evt):
        """Handle window close event"""
        self.window_closed = True
        plt.close('all')
        exit()  # Terminate the program

    # Load the maze from a text file
    def loadMaze(self, file):
        maze = []
        with open(file, 'r') as f:
            for line in f:
                row = line.strip().split()
                maze.append(row)
        
        # Convert to integers
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                maze[i][j] = int(maze[i][j])
        
        print(f"Loaded maze of size {len(maze)}x{len(maze[0])}")
        return maze

    # Manhattan distance heuristic
    def heuristic(self, x, y):
        # Use the updated h-values for Adaptive A* if available, else use the Manhattan distance
        if (x, y) in self.h_values:
            return self.h_values[(x, y)]
        dx = abs(x - self.goal[0])
        dy = abs(y - self.goal[1])
        return dx + dy

    # Get valid neighbors (N, S, E, W)
    def getNeighbors(self, x, y):
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Up, Down, Left, Right
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:  # Stay within bounds
                neighbors.append((nx, ny))
        return neighbors

    # Draw the current state of the maze
    def drawMaze(self, current_pos=None, path=None, delay=0.05):
        # Create display grid
        displayGrid = np.zeros((self.size, self.size), dtype=int)
        
        # Fill in knowledge
        for i in range(self.size):
            for j in range(self.size):
                if self.knowledge[i][j] == '1':
                    displayGrid[i][j] = 1  # Blocked
                elif self.knowledge[i][j] == '0':
                    displayGrid[i][j] = 0  # Unblocked & known
                elif self.knowledge[i][j] == 'P':
                    displayGrid[i][j] = 2  # Path
                elif self.grid[i][j] == 1:
                    displayGrid[i][j] = 9 #Blocked and unknown
                else:
                    displayGrid[i][j] = 3  # Unknown
        
        # Show current planned path if provided
        if path:
            for x, y in path:
                if displayGrid[x][y] != 1:  # Don't overwrite blocked cells
                    displayGrid[x][y] = 4  # Path being planned
        
        # Mark visited cells
        for x, y in self.visited_cells:
            if displayGrid[x][y] == 0:  # Only mark if it's a known unblocked cell
                displayGrid[x][y] = 7  # Visited
        
        # Mark current position if provided
        if current_pos:
            displayGrid[current_pos[0]][current_pos[1]] = 5  # Current position
        
        # Mark start and goal
        displayGrid[self.start[0]][self.start[1]] = 6  # Start
        displayGrid[self.goal[0]][self.goal[1]] = 8  # Goal
        
        # Define custom colormap with additional colors
        cmap = matplotlib.colors.ListedColormap(['white', 'black', 'lightgreen', '#D3D3D3', 
                                            'yellow', 'blue', 'green', 'lightblue', 'red', '#A9A9A9'])
        
        # Clear and redraw
        self.ax.clear()
        self.ax.imshow(displayGrid, cmap=cmap, interpolation='nearest')
        self.ax.set_xticks([]) 
        self.ax.set_yticks([])
        self.ax.set_title(f'Adaptive A* - Replans: {self.replans} - Expanded: {self.cells_expanded}')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Known Unblocked'),
            Patch(facecolor='black', edgecolor='gray', label='Blocked'),
            Patch(facecolor='lightgreen', edgecolor='gray', label='Final Path'),
            Patch(facecolor='lightgray', edgecolor='gray', label='Unknown'),
            Patch(facecolor='yellow', edgecolor='gray', label='Planned Path'),
            Patch(facecolor='blue', edgecolor='gray', label='Current Position'),
            Patch(facecolor='green', edgecolor='gray', label='Start'),
            Patch(facecolor='lightblue', edgecolor='gray', label='Visited'),
            Patch(facecolor='red', edgecolor='gray', label='Goal')
        ]
        self.ax.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='small')
        
        plt.pause(delay)  # Short pause for animation
        plt.draw()

    # Reveal neighboring cells around current position
    def senseNeighbors(self, x, y):
        has_updated = False
        # Examine all neighboring cells
        for nx, ny in self.getNeighbors(x, y):
            # Check if cell was unknown before
            if self.knowledge[nx][ny] == 'U':
                # Update knowledge based on actual maze
                if self.grid[nx][ny] == 1:
                    self.knowledge[nx][ny] = '1'  # Blocked
                else:
                    self.knowledge[nx][ny] = '0'  # Unblocked
                has_updated = True
        return has_updated

    # A* Search with Replanning (Modified for Adaptive A*)
    def search(self):
        # Initialize the agent's position and path
        current_pos = self.start
        self.visited_cells.add(current_pos)
        final_path = [current_pos]
        steps_taken = 0
        
        # Main loop - continue until goal is reached or no path possible
        while current_pos != self.goal:
            steps_taken += 1
            
            if self.senseNeighbors(*current_pos):
                # Update knowledge, possibly replan
                self.replans+=1 #increment the number of replans
            # Sense neighboring cells
            self.senseNeighbors(*current_pos)
            
            # Plan path from current position to goal
            planned_path = self.planPath(current_pos)
            
            # Visualize current state and plan
            self.drawMaze(current_pos, planned_path)
            
            if not planned_path or len(planned_path) <= 1:
                print("No path to goal found from current position")
                print(f"Total Cells Expanded: {self.cells_expanded}")
                print(f"Total Replans: {self.replans}")
                plt.show(block=True)
                return None
            
            # Move to next position on planned path
            next_pos = planned_path[1]  # Skip current position in path
            
            # Check if next position is actually traversable
            if self.grid[next_pos[0]][next_pos[1]]==1:
                # Update knowledge if we encounter a blocked cell
                self.knowledge[next_pos[0]][next_pos[1]] = '1'
                self.replans+=1 #increment the number of replans
                # Don't move, replan in next iteration
                continue
            
            # Move to next position
            current_pos = next_pos
            self.visited_cells.add(current_pos)
            final_path.append(current_pos)
            
            # Mark current position as known unblocked
            self.knowledge[current_pos[0]][current_pos[1]] = '0'
        
        # Goal reached
        print(f"Goal reached in {steps_taken} steps!")
        
        # Mark the final path
        for x, y in final_path:
            self.knowledge[x][y] = 'P'
        
        # Final visualization
        self.drawMaze(current_pos, None)
        plt.show(block=True)
        

        return final_path

    # Plan a path from start to goal using A* (Modified for Adaptive A*)
    def planPath(self, start):
        open_heap = BinaryHeap()
        closed_set = set()
        g_scores = {start: 0}
        f_scores = {start: self.heuristic(*start)}
        parents = {}

        c = 5000  # Example constant, adjust as needed

        # Breaking ties for larger g-values
        if self.tie_break == 'large':
            open_heap.insert((c * f_scores[start] - g_scores[start], start))
        else:
            open_heap.insert((f_scores[start] + c * g_scores[start], start))


        while not open_heap.isEmpty():
            # Get node with lowest priority
            f, current = open_heap.delMin()
            # Skip if already processed
            if current in closed_set:
                continue
                
            self.cells_expanded+=1   #increment the number of cells expanded
            # Mark as processed
            closed_set.add(current)
            
            # If we reached the goal, reconstruct the path
            if current == self.goal:
                path = self.reconstructPath(parents, start, self.goal)
                if path and path[-1] == self.goal:
                    # Update h-values based on the expanded states
                    for node in closed_set:
                        g_val = g_scores[node]
                        self.h_values[node] = g_scores[self.goal] - g_val
                    return path
                else:
                    print(f"Path does not reach the goal. Path found: {path}")
                    return None

            # Explore neighbors
            for neighbor in self.getNeighbors(*current):
                nx, ny = neighbor

                # Skip if known to be blocked or already processed
                if self.knowledge[nx][ny] == '1' or neighbor in closed_set:
                    continue

                # Calculate tentative g score
                tentative_g = g_scores[current] + 1

                # If better path found, update scores
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    parents[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + self.heuristic(*neighbor)

                    # Insert updated node into the heap with adjusted priority
                    open_heap.insert((c * f_scores[neighbor] - g_scores[neighbor], neighbor))

        # No path found
        return None

    # Reconstruct the path from parents dictionary
    def reconstructPath(self, parents, start, goal):
        path = [goal]
        current = goal
        
        while current in parents and current != start:
            current = parents[current]
            path.append(current)
        
        path.reverse()
        
        # If start is not in the path, prepend it
        if path[0] != start:
            path.insert(0, start)
            
        return path

# Example Usage
if __name__ == '__main__':
    mazeFile = 'mazes/maze_4.txt'  # Change this to test different mazes
    adaptiveAStar = AdaptiveAStar(mazeFile)
    path = adaptiveAStar.search()
    
    if path:
        print("Complete Path Found!")
        print("Path Length:", len(path))
    else:
        print("No Path Found.")
