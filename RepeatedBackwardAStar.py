import random
import matplotlib
matplotlib.use('TkAgg')  # Ensures the right backend for interactive windows
import matplotlib.pyplot as plt
import numpy as np
import time

from BinaryHeap import BinaryHeap

class RepeatedBackwardAStar:
    def __init__(self, mazeFile):
        self.grid = self.loadMaze(mazeFile)  # Load maze from file
        self.size = len(self.grid)  # Size of the grid (Assuming square grid)
        self.start = (self.size - 1, self.size - 1)  # Starting position (top-left corner)
        self.goal = (0,0)  # Goal position (bottom-right corner)
        # Sense neighbors of start and goal immediately
       

        # Force start and goal to be unblocked in the actual grid
        self.grid[self.start[0]][self.start[1]] = 0  # Force start to be unblocked
        self.grid[self.goal[0]][self.goal[1]] = 0    # Force goal to be unblocked

        self.knowledge = [['U' for _ in range(self.size)] for _ in range(self.size)]  # Agent's knowledge of the grid
        self.knowledge[self.start[0]][self.start[1]] = '0'  # Start is known to be unblocked
        self.knowledge[self.goal[0]][self.goal[1]] = '0'    # Goal is assumed to be unblocked
        
        # Setup for dynamic visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Repeated Backward A* Maze Solver')
        self.visited_cells = set()  # Keep track of visited cells
        # Add window close event handler
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
    
        # Add flag to track if window was closed
        self.window_closed = False

        # Performance metrics
        self.cells_expanded = 0
        self.replans = 0
        self.senseNeighbors(self.start[0], self.start[1])
        self.senseNeighbors(self.goal[0], self.goal[1])

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
        
        # Convert strings to integers
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                maze[i][j] = int(maze[i][j])
        
        print(f"Loaded maze of size {len(maze)}x{len(maze[0])}")
        return maze

    # Manhattan distance heuristic - from position to start (reversed)
    def heuristic(self, x, y):
        return abs(x - self.goal[0]) + abs(y - self.goal[1])

    # Get valid neighbors (N, S, E, W)
    def getNeighbors(self, x, y):
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Down, Left, Up (priority order)
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
                    displayGrid[i][j] = 2  # Final Path
                elif self.grid[i][j] == 1:
                    displayGrid[i][j] = 9 #Blocked and unknown
                else:
                    displayGrid[i][j] = 3  # Unknown
        
        # Show current planned path if provided
        if path and current_pos!=self.goal:
            for x, y in path:
                if (x, y) != current_pos and (x, y) != self.start and (x, y) != self.goal and displayGrid[x][y] != 1:
                    displayGrid[x][y] = 4  # Path being planned
        
        # Mark visited cells
        for x, y in self.visited_cells:
            if displayGrid[x][y] == 0 and (x, y) != current_pos and (x, y) != self.start and (x, y) != self.goal:
                displayGrid[x][y] = 7  # Visited
        
        # Mark start and goal
        displayGrid[self.start[0]][self.start[1]] = 6  # Start
        displayGrid[self.goal[0]][self.goal[1]] = 8  # Goal
        
        # Mark current position if provided
        if current_pos and current_pos != self.start and current_pos != self.goal:
            displayGrid[current_pos[0]][current_pos[1]] = 5  # Current position
        
        # Define custom colormap with additional colors
        cmap = matplotlib.colors.ListedColormap(['white', 'black', 'lightgreen', 'lightgrey', 
                                            'yellow', 'blue', 'green', 'lightblue', 'red', '#A9A9A9'])
        
        # Clear and redraw
        self.ax.clear()
        self.ax.imshow(displayGrid, cmap=cmap, interpolation='nearest')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(f'Repeated Backward A* - Replans: {self.replans} - Expanded: {self.cells_expanded}')
        
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
        
        plt.pause(delay)  # Pause for animation
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
                    self.replans += 1
                else:
                    self.knowledge[nx][ny] = '0'  # Unblocked
                has_updated = True
        return has_updated

    # A* Search with Replanning
    def search(self):
        # Initialize the agent's position and path
        current_pos = self.start
        self.visited_cells.add(current_pos)
        self.senseNeighbors(*current_pos)
        final_path = [current_pos]
        steps_taken = 0
        
        # Main loop - continue until goal is reached or no path possible
        while current_pos != self.goal:
            steps_taken += 1

            # Sense neighboring cells
            self.senseNeighbors(*current_pos)
            
            # Plan path from current position to goal using Backward A*
            planned_path = self.planBackwardPath(current_pos)
            
            # Visualize current state and plan
            self.drawMaze(current_pos, planned_path)
            
            if not planned_path or len(planned_path) <= 1:
                print(f"No path to goal found from current position {current_pos}")
                plt.show(block=True)
                return None
            
            # Move to next position on planned path
            next_pos = planned_path[1]  # Skip current position in path
            
            # Check if next position is actually traversable
            if self.grid[next_pos[0]][next_pos[1]] == 1:
                # Update knowledge if we encounter a blocked cell
                """"Unexpected obstacle at {next_pos}, updating knowledge"""
                self.knowledge[next_pos[0]][next_pos[1]] = '1'
                self.replans += 1
                # Don't move, replan in next iteration
                continue
            
            # Move to next position
            current_pos = next_pos
            self.visited_cells.add(current_pos)
            final_path.append(current_pos)
            
            # Mark current position as known unblocked
            self.knowledge[current_pos[0]][current_pos[1]] = '0'
            
            # Brief pause to better visualize movement
            time.sleep(0.1)
        
        # Goal reached
        print(f"Goal reached in {steps_taken} steps!")
        print(f"Path length: {len(final_path)}")
        print(f"Cells expanded: {self.cells_expanded}")
        print(f"Replans: {self.replans}")
        
        # Mark the final path
        for x, y in final_path:
            self.knowledge[x][y] = 'P'
        
        # Final visualization with longer delay
        self.drawMaze(current_pos, None)
        plt.show(block=True)
        
        return final_path

    # Plan a path from goal to current position using Backward A*
    def planBackwardPath(self, current):
        # Reset cells expanded counter for this planning session
        local_cells_expanded = 0
        # In Backward A*, we search from goal to current position
        start_search = current
        goal_search = self.goal
        
        # Debug information
        #print(f"Planning path from goal {start_search} to current {goal_search}")
        
        open_heap = BinaryHeap()
        closed_set = set()
        g_scores = {start_search: 0}
        f_scores = {}
        parents = {}
        
        # Initialize scores for start node (the goal in backward search)
        f_scores[start_search] = self.heuristic(*start_search)  # Heuristic to start position
        
        # Insert start node into open list
        open_heap.insert((f_scores[start_search], start_search))
        
        # Add a safety counter to prevent infinite loops
        
        while not open_heap.isEmpty():
            
            
            # Get node with lowest f-score
            current_f, current_node = open_heap.delMin()
            
            # Skip if already processed
            if current_node in closed_set:
                continue
                
            local_cells_expanded += 1
            # Mark as processed
            closed_set.add(current_node)
            
            # If we reached the goal (current agent position in backward search), reconstruct the path
            if current_node == goal_search:
                # Increment global cells expanded counter
                self.cells_expanded += local_cells_expanded
                # Reconstruct path and reverse it (since we searched backward)
                path = self.reconstructPath(parents, start_search, goal_search)
                #print(f"Path found! Length: {len(path) if path else 'N/A'}")
                return path
            
            
            
            
            
            # Explore neighbors
            for neighbor in sorted(self.getNeighbors(*current_node),
                                   key=lambda n: (self.knowledge[n[0]][n[1]] != '0', self.heuristic(*n))):
                if self.knowledge[neighbor[0]][neighbor[1]] == '1' or neighbor in closed_set:
                    continue
                
                # Skip if known to be blocked or already processed
                #if self.knowledge[nx][ny] == '1' or neighbor in closed_set:
                   # continue
                
                # Calculate tentative g score
                tentative_g = g_scores[current_node] + 1
                
                # Add neighbor to open list or update it if we found a better path
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    parents[neighbor] = current_node
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + self.heuristic(*neighbor)
                    open_heap.insert((f_scores[neighbor], neighbor))

                    
        
        # If we reach here, either the heap is empty or we exceeded max iterations
        
        
        
        # No path found
        return None
    # Reconstruct the path from parents dictionary
    def reconstructPath(self, parents, start, goal):
            path = [goal]
            current = goal
            
            while current in parents:
                current = parents[current]
                path.append(current)
            
            # Reverse the path since we searched backward
            path.reverse()
            
            # The path should now go from agent's current position to goal
            if path[0] != start:
                # Something went wrong - should start at goal
                return None
                
            return path

# Example Usage
if __name__ == '__main__':
    mazeFile = 'mazes/maze_0.txt'  # Change this to test different mazes
    rbAStar = RepeatedBackwardAStar(mazeFile)
    path = rbAStar.search()
    
