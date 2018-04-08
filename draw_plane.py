import numpy as np
import time
import sys
if sys.version_info.major == 2:
	import Tkinter as tk
else:
	import tkinter as tk

class DrawPlane(tk.Tk, object):
	def __init__(self, width, height, w_offset, h_offset, scale_factor=1):
		super(DrawPlane, self).__init__()
		self.canvas = tk.Canvas(self, 
								bg='magenta', 
								height=(height - h_offset) * scale_factor, 
								width= (width - w_offset) * scale_factor)
		self.w_offset = w_offset
		self.h_offset = h_offset
		self.scale_factor = scale_factor
		# self.geometry('500x300')

	def remove_offset(self, loc):
		return np.array(loc) - np.array((self.w_offset, self.h_offset))

	def draw_cell(self, center, radius, type='NUMB'):
		center = self.remove_offset(center) * self.scale_factor
		radius = radius * self.scale_factor
		cell = self.canvas.create_oval(
			center[0] - radius,
			center[1] - radius,
			center[0] + radius,
			center[1] + radius)
		if type == 'AI':
			fill_color="red"
		elif type == 'NUMB':
			fill_color="green"
		elif type == 'STATE':
			fill_color = "yellow"
		self.canvas.itemconfig(cell, fill=fill_color)
		return cell
	def draw_target(self, target, cell_radius, tolerance):
		target = self.remove_offset(target) * self.scale_factor
		cell_radius = cell_radius * self.scale_factor
		tolerance = tolerance * self.scale_factor
		self.canvas.create_oval(
						target[0] - cell_radius - tolerance,
						target[1] - cell_radius - tolerance,
						target[0] + cell_radius + tolerance,
						target[1] + cell_radius + tolerance)


if __name__ == '__main__':
	d = DrawPlane(200, 200)