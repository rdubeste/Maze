import numpy as np
import pyglet
import colorsys
from enum import Enum
import heapq

_rows = 0
_cols = 0
frontier = []
maze = []
_c_size = 0
_color_cycle = 0
batch = pyglet.graphics.Batch()
random = np.random
window = pyglet.window.Window(10, 10, "maze", resizable=False)
_color_mode = None
_gen_mode = None


class ColorMode(Enum):
    PATH = 0
    HSV = 1
    FRONTIER = 2


class GenMode(Enum):
    SPANNING = 0
    RADIAL = 1


class Cell:

    state = None
    added = None
    coord = None
    weight = 0
    dist = 0

    def __init__(self, i, j, weight):
        self.state = 0
        self.added = self
        self.coord = (i, j)
        self.weight = weight


def start(rows, cols, c_size=7, color_cycle=-1, color_mode=ColorMode.PATH, gen_mode=GenMode.SPANNING):
    global _rows, _cols, _c_size, window, _color_cycle, _color_mode, _gen_mode

    # maze constants
    _rows = rows
    _cols = cols
    _c_size = c_size
    _color_mode = color_mode
    _gen_mode = gen_mode

    if color_cycle == -1:
        _color_cycle = (rows + cols) * 0.75
    else:
        _color_cycle = color_cycle

    # maze setup
    init_maze()

    # start generation
    window.set_size((_rows + 1) * _c_size, (_cols + 1) * _c_size)
    pyglet.clock.schedule(update, 0.01)
    pyglet.app.run()


def init_maze():
    global maze, frontier

    # create maze grid
    maze = np.full((_rows, _cols), 0, dtype=Cell)

    for row in range(0, _rows):
        for col in range(0, _cols):
            maze[row][col] = Cell(row, col, random.random())

    # add starting cell to frontier
    if _gen_mode == GenMode.SPANNING:
        heapq.heappush(frontier, (maze[1][1].weight, maze[1][1]))
    elif _gen_mode == GenMode.RADIAL:
        frontier.append(maze[1][1])


@window.event
def on_draw():
    global batch
    window.clear()
    batch.draw()


def update(_, __):

    # while there are cells in the frontier list, add a new cell to the maze
    if len(frontier) > 0:
        step()
    else:
        # draw_final()
        pass


def add_to_batch(cell: Cell):
    global batch, _color_mode
    if _color_mode != ColorMode.FRONTIER:
        if cell.state == 2:
            return
    (x, y) = cell.coord
    true_x = x * _c_size
    true_y = y * _c_size
    if _color_mode == ColorMode.PATH:
        color = (255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255)
    elif _color_mode == ColorMode.HSV:
        rgb_color = colorsys.hsv_to_rgb(cell.dist / _color_cycle, 1, 1)
        color = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255)) * 4
    elif _color_mode == ColorMode.FRONTIER:
        if cell.state == 2:
            color = (255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0)
        else:
            color = (255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255)

    batch.add(4, pyglet.gl.GL_QUADS, None,
              ('v2f', (true_x, true_y,
                       true_x + _c_size, true_y,
                       true_x + _c_size, true_y + _c_size,
                       true_x, true_y + _c_size)),
              ('c3B', color))


def draw_final():
    global maze, batch
    batch = pyglet.graphics.Batch()
    for row in range(0, _rows):
        for col in range(0, _cols):
            if maze[row][col].state == 1:
                (x, y) = maze[row][col].coord
                true_x = x * _c_size
                true_y = y * _c_size
                batch.add(4, pyglet.gl.GL_QUADS, None,
                          ('v2f', (true_x, true_y,
                                   true_x + _c_size, true_y,
                                   true_x + _c_size, true_y + _c_size,
                                   true_x, true_y + _c_size)),
                          ('c3B', (255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255)))


def step():
    global batch, _gen_mode

    cell = Cell
    if _gen_mode == GenMode.RADIAL:
        # pick a random grid cell in the list of frontier cells and add it to the maze
        cell = frontier[random.randint(0, len(frontier))]
        frontier.remove(cell)
    elif _gen_mode == GenMode.SPANNING:
        cell = heapq.heappop(frontier)[1]

    (i, j) = cell.coord
    cell.state = 1
    cell.added.state = 1

    # add new cells to the drawing batch
    add_to_batch(cell)
    add_to_batch(cell.added)

    # find adjacent empty cells and add them to the frontier
    if i + 2 < _rows:
        adj = maze[i + 2][j]
        if adj.state == 0:
            mark(adj, cell, i + 1, j)
    if j + 2 < _cols:
        adj = maze[i][j + 2]
        if adj.state == 0:
            mark(adj, cell, i, j + 1)
    if i - 2 > 0:
        adj = maze[i - 2][j]
        if adj.state == 0:
            mark(adj, cell, i - 1, j)
    if j - 2 > 0:
        adj = maze[i][j - 2]
        if adj.state == 0:
            mark(adj, cell, i, j - 1)


def mark(cell: Cell, prev: Cell, i, j):
    cell.state = 2
    cell.added = maze[i][j]
    cell.added.state = 2
    cell.added.dist = prev.dist + 1
    cell.dist = prev.dist + 2

    add_to_batch(cell.added)
    if _gen_mode == GenMode.SPANNING:
        heapq.heappush(frontier, (cell.weight, cell))
    elif _gen_mode == GenMode.RADIAL:
        frontier.append(cell)

# Start Everything
start(120, 120, 8, gen_mode=GenMode.RADIAL, color_mode=ColorMode.HSV)
