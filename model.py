# model.py
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# import matplotlib.pyplot as plt
import random
import sys,time
import numpy as np
import operator

from draw_plane import DrawPlane
SAVE_FIG = False
RUN_LEARNING = True

START_POINT = 168
END_POINT = 190
TICK_RESOLUTION = 10
AI_CELL = 'Cpaaa'

PLANE_THRESHOLD = 3			#draw the cells in the [-2,+2] planes (5 total)
PLANE_DRAW = 7
CANVAS_WIDTH = 310
CANVAS_HEIGHT = 370
CANVAS_WIDTH_OFFSET = 140
CANVAS_HEIGHT_OFFSET = 180
CANVAS_DISPLAY_SCALE_FACTOR = 2		### zoom in/out the size of canvas
FRESH_TIME = 0.02
FRESH_PERIOD = 1

#Here used manually adjusted data
EMBRYO_LONG_AXIS_1 = [162, 194]
EMBRYO_LONG_AXIS_2 = [293, 351]
EMBRYO_SHORT_AXIS_1 = [178, 317]
EMBRYO_SHORT_AXIS_2 = [274, 232]

PLANE7_LONG_AXIS_1 = [170, 205]
PLANE7_LONG_AXIS_2 = [283, 333]
PLANE7_SHORT_AXIS_1 = [190, 303]
PLANE7_SHORT_AXIS_2 = [263, 237]

PLANE7_CENTER = [225, 275]
PLANE7_VEC0 = [1, 1]
PLANE7_VEC1 = [1, -1]
PLANE7_E0 = np.linalg.norm(np.array(PLANE7_LONG_AXIS_2) - np.array(PLANE7_CENTER))
PLANE7_E1 = np.linalg.norm(np.array(PLANE7_SHORT_AXIS_2) - np.array(PLANE7_CENTER))

EMBRYO_TOTAL_PLANE = 30
PLANE_RESOLUTION = 5

RADIUS_SCALE_FACTOR = 1.0     ####if manual adjustment needed, because radii are estimated.

#State cell list find from the Cell Neighbor Dertermination model
STATE_CELL_LIST = ['ABarpppap', 'ABarppppa', 'ABarppppp', 'Caaaa', 'ABprapapp', 'Epra', 'ABprapaaa', \
				'ABprapaap', 'Cpaap', 'ABprapapa', 'ABarppapp', 'Caaap', 'Eprp', 'ABarpppaa', 'Eplp', \
				'ABarppapa', 'Epla', 'ABarppaap']

AI_CELL_SPEED_PER_MIN = 2

AI_TARGET = [248, 299, 7.0*5]
AI_CELL_TARGET_TOLERANCE = 1
AI_CELL_BEGIN_REWARD = 20
class SeqRosModel(Model):
	def __init__(self):
		self.file_path = './nuclei/t%03d-nuclei'
		self.start_point = START_POINT
		self.end_point = END_POINT
		self.ai_cell = AI_CELL
		self.ai_cell_target = np.array(AI_TARGET)
		self.ai_cell_target_tolerance = AI_CELL_TARGET_TOLERANCE
		self.ticks = 0
		self.tick_resolution = TICK_RESOLUTION
		self.end_tick = (self.end_point - self.start_point) * self.tick_resolution
		self.stage_destination_point = self.start_point

		self.embryo_long_axis = np.linalg.norm(np.array(EMBRYO_LONG_AXIS_2) - np.array(EMBRYO_LONG_AXIS_1))
		self.embryo_short_axis = np.linalg.norm(np.array(EMBRYO_LONG_AXIS_2) - np.array(EMBRYO_LONG_AXIS_1))
		self.embryo_total_plane = EMBRYO_TOTAL_PLANE
		self.plane_resolution = PLANE_RESOLUTION
		self.embryo_volume = 4 / 3.0 * np.pi \
						* (0.5 * self.embryo_long_axis) \
						* (0.5 * self.embryo_short_axis) \
						* (0.5 * self.embryo_total_plane * self.plane_resolution)
		self.radius_scale_factor = RADIUS_SCALE_FACTOR


		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}
		self.state_cell_list = STATE_CELL_LIST
		self.state_value_dict = {}
		self.schedule = RandomActivation(self)

		self.init_env()
		self.update_stage_destination()

		self.plane = DrawPlane(width=CANVAS_WIDTH, 
							height=CANVAS_HEIGHT, 
							w_offset = CANVAS_WIDTH_OFFSET,
							h_offset = CANVAS_HEIGHT_OFFSET,
							scale_factor = CANVAS_DISPLAY_SCALE_FACTOR)
		self.canvas = self.plane.canvas
		self.plane_draw = PLANE_DRAW
		self.draw(self.plane_draw)


		########## Reinforcement Learning related ##############
		self.ai_cell_speed = AI_CELL_SPEED_PER_MIN / float(TICK_RESOLUTION)
		self.n_observations = (len(self.state_cell_list) + 1) * 3
		self.actions = ['n','s','w','e','nw','ne','sw','se']
		self.n_actions = len(self.actions)

	def dist_point_ellipse(self, old_point, origin=PLANE7_CENTER, direc_vec0=PLANE7_VEC0, direc_vec1=PLANE7_VEC1):
		origin = np.array(origin)
		old_point = np.array(old_point)
		direc_vec0 = np.array(direc_vec0)
		direc_vec1 = np.array(direc_vec1)

		vec = old_point - origin
		vec_norm = np.linalg.norm(vec)
		direc_vec0_norm = np.linalg.norm(direc_vec0)
		direc_vec1_norm = np.linalg.norm(direc_vec1)
		cos_vec0 = vec.dot(direc_vec0) / vec_norm / direc_vec0_norm
		cos_vec1 = vec.dot(direc_vec1) / vec_norm / direc_vec1_norm
		x0 = vec_norm * cos_vec0
		x1 = vec_norm * cos_vec1
		new_point = np.array((x0,x1))

		new_point = np.abs(new_point)

		distance = -1.0
		if new_point[1] > 0:
			if new_point[0] > 0:
				###compute the unique root tbar of F(t) on (-e1*e1, +inf)
				t1 = - PLANE7_E1 * PLANE7_E1
				t2 = 1000000
				while True:
					ft_mid = (PLANE7_E0 * new_point[0] / (0.5 * (t1 + t2) + PLANE7_E0 ** 2)) ** 2 \
							+(PLANE7_E1 * new_point[1] / (0.5 * (t1 + t2) + PLANE7_E1 ** 2)) ** 2 - 1
					if ft_mid > 0:
						t1 = 0.5 * (t1 + t2)
					elif ft_mid < 0:
						t2 = 0.5 * (t1 + t2)
					else:
						break
					if np.abs(t1-t2)<0.001:
						break
				tbar = 0.5 * (t1 + t2)
				x0 = PLANE7_E0 * PLANE7_E0 * new_point[0] / (tbar + PLANE7_E0 * PLANE7_E0)
				x1 = PLANE7_E1 * PLANE7_E1 * new_point[1] / (tbar + PLANE7_E1 * PLANE7_E1)
				distance = np.sqrt((x0 - new_point[0]) ** 2 + (x1 - new_point[1]) ** 2)
			else:
				x0 = 0
				x1 = PLANE7_E1
				distance = np.abs(new_point[1] - PLANE7_E1)
		else:
			if new_point[0] < (PLANE7_E0 ** 2 - PLANE7_E1 ** 2) / PLANE7_E0:
				x0 = PLANE7_E0 * PLANE7_E0 * new_point[0] / (PLANE7_E0 ** 2 - PLANE7_E1 ** 2)
				x1 = PLANE7_E1 * np.sqrt(1 - (x0 / PLANE7_E0) ** 2)
				distance = np.sqrt((x0 - new_point[0]) ** 2 + x1 ** 2)
			else:
				x0 = PLANE7_E0
				x1 = 0
				distance = np.abs(new_point[0] - PLANE7_E0)

		return distance



	def draw(self,n_plane):
		self.canvas.delete("all")
		draw_range = np.arange(n_plane-PLANE_THRESHOLD, n_plane+PLANE_THRESHOLD+1, 1)
		draw_range = draw_range.tolist()
		draw_range.reverse()

		for n in draw_range:
			angle = np.pi *0.5 / (PLANE_THRESHOLD + 1) * np.abs(n - n_plane)
			for cell in self.schedule.agents:
				if cell.cell_name == self.ai_cell:
					type = 'AI'
					self.plane.draw_target(self.ai_cell_target[0:2], cell.diameter/2.0, self.ai_cell_target_tolerance)
				elif cell.cell_name in self.state_cell_list:
					type = 'STATE'
				else:
					type = 'NUMB'
				if round(cell.location[2]) == n:
					self.plane.draw_cell(center=cell.location[0:2], 
										radius=cell.diameter/2.0*np.cos(angle),
										type=type)

		self.canvas.pack()
		self.canvas.update()
		time.sleep(FRESH_TIME)

	def get_radius(self, cell_name):
		if cell_name[0:2]=="AB":
			v=0.55*(0.5**(len(cell_name)-2))
		elif cell_name=="P1":
			v=0.45
		elif cell_name=="EMS":
			v=0.45*0.54
		elif cell_name=="P2":
			v=0.45*0.46
		elif cell_name[0:2]=="MS":
			v=0.45*0.54*0.5*(0.5**(len(cell_name)-2))
		elif cell_name=="E":
			v=0.45*0.54*0.5
		elif cell_name[0]=="E" and len(cell_name)>=2 and cell_name[1] != "M":
			v=0.45*0.54*0.5*(0.5**(len(cell_name)-1))
		elif cell_name[0]=="C":
			v=0.45*0.46*0.53*(0.5**(len(cell_name)-1))
		elif cell_name=="P3":
			v=0.45*0.46*0.47
		elif cell_name[0]=="D":
			v=0.45*0.46*0.47*0.52*(0.5**(len(cell_name)-1))
		elif cell_name=="P4":
			v=0.45*0.46*0.47*0.48
		else:
			print('ERROR!!!!! CELL NOT FOUND IN CALCULATING HER RADIUS!!!!')

		radius = pow(self.embryo_volume * v / (4 / 3.0 * np.pi), 1/3.0)
		radius = radius * self.radius_scale_factor

		return radius
		


	def get_cell_daughter(self, cell_name, cell_dict):
		daughter = []
		if cell_name == 'P0':
			daughter = ['AB', 'P1']
		elif cell_name == 'P1':
			daughter = ['EMS', 'P2']
		elif cell_name == 'P2':
			daughter = ['C', 'P3']
		elif cell_name == 'P3':
			daughter = ['D', 'P4']
		elif cell_name == 'P4':
			daughter = ['Z2', 'Z3']
		elif cell_name == 'EMS':
			daughter = ['MS', 'E']
		## standard name ###
		else:
			for cell in cell_dict.keys():
				if cell.startswith(cell_name) and len(cell) == len(cell_name) + 1:
					daughter.append(cell)
			daughter = sorted(daughter)
		if daughter == []:
			daughter = ['', '']
		return daughter

	def init_env(self):
		with open(self.file_path % self.start_point) as file:
			for line in file:
				line = line[:len(line)-1]
				vec = line.split(', ')
				id = int(vec[0])
				location = np.array((float(vec[5]), float(vec[6]), float(vec[7])))
				diameter = float(vec[8])
				cell_name = vec[9]
				if cell_name != '':
					self.current_cell_list.append(cell_name)
					a = CellAgent(id, self, cell_name, location, diameter)
					self.schedule.add(a)

	def set_cell_next_location(self,ai_action):
		for cell in self.schedule.agents:
			if cell.cell_name in self.next_stage_destination_list:
				cell.next_location = (self.next_stage_destination_list[cell.cell_name][0:3] - cell.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + cell.location
				cell.diameter = self.next_stage_destination_list[cell.cell_name][3]
			else:
				### new cell born ###
				mother = cell.cell_name
				daughter = self.get_cell_daughter(cell.cell_name, self.next_stage_destination_list)
				if daughter[0] == '':
					print('ERROR!!!!! NO DAUGHTER FOUND!!!!!')
				cell.cell_name = daughter[0]
				cell.diameter = self.next_stage_destination_list[daughter[0]][3]
				cell.next_location = (self.next_stage_destination_list[daughter[0]][0:3] - cell.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + cell.location
				new_id = len(self.schedule.agents) + 1
				new_diameter = self.next_stage_destination_list[daughter[1]][3]
				a = CellAgent(new_id, self, daughter[1], cell.location, new_diameter)
				self.schedule.add(a)
				a.next_location = (self.next_stage_destination_list[daughter[1]][0:3] - a.location) \
								/ (self.tick_resolution - self.ticks % self.tick_resolution) + a.location
				
				self.dividing_cell_overall.append(mother)

			if RUN_LEARNING and cell.cell_name == self.ai_cell:
				offset = self.get_ai_next_location_offset(ai_action)
				cell.next_location[0:2] = cell.location[0:2] + offset[0:2]

	def update_stage_destination(self):
		current_stage_destination_point = self.start_point + 1 + int(self.ticks / self.tick_resolution)
		if self.stage_destination_point == current_stage_destination_point:
			return
		else:
			self.stage_destination_point = current_stage_destination_point
			self.next_stage_destination_list.clear()
			with open(self.file_path % self.stage_destination_point) as file:
				for line in file:
					line = line[:len(line)-1]
					vec = line.split(', ')
					id = int(vec[0])
					loc_and_dia = np.array((float(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])))
					cell_name = vec[9]
					if cell_name != '':
						self.next_stage_destination_list[cell_name] = loc_and_dia

	def render(self):
		if self.ticks % FRESH_PERIOD == 0:
			self.draw(self.plane_draw)

	def reset(self):
		self.ticks = 0
		self.stage_destination_point = self.start_point
		self.current_cell_list = []
		self.dividing_cell_overall = []
		self.next_stage_destination_list = {}
		self.state_value_dict = {}
		del self.schedule.agents[:]

		self.init_env()
		self.update_stage_destination()

		s = self.get_state()

		return s

	def get_state(self):
		s = np.round(self.state_value_dict[self.ai_cell], decimals=4).tolist()
		sorted_state_value_dict = sorted(self.state_value_dict.items(), key=operator.itemgetter(0))
		for item in sorted_state_value_dict:
			if item[0] != self.ai_cell:
				s = s + np.round(item[1], decimals=4).tolist()
		for i in range(0,len(s)):
			if i % 3 == 2:
				s[i] *= self.plane_resolution

		return s

	def get_reward(self):
		r = 0
		done = False
		ai_location = np.copy(self.state_value_dict[self.ai_cell])
		ai_location[2] = ai_location[2] * self.plane_resolution
		ai2bdr_dist = self.dist_point_ellipse(ai_location[0:2])
		ai_radius = self.get_radius(self.ai_cell)

		dist2target = np.linalg.norm(self.ai_cell_target - ai_location)
		if dist2target < AI_CELL_BEGIN_REWARD:
			r = (AI_CELL_BEGIN_REWARD -  dist2target)
			if dist2target < self.ai_cell_target_tolerance:
				r = 20

				return r, done
			

		#########boundary control rule: (1) dist>0.8*radius, ok  (2)0.5*r<dist<0.8*r, bad    (3)dist<0.5*r, dead
		if ai2bdr_dist > 0.8 * ai_radius:
			r += 0
		elif ai2bdr_dist > 0.5 * ai_radius and ai2bdr_dist <= 0.8 * ai_radius:
			r += (0.8 - float(ai2bdr_dist) / ai_radius) / (0.5 - 0.8)		## 0.5->-1, 0.8->0
		elif ai2bdr_dist < 0.5 * ai_radius:
			r = -10
			done = True
			print('hit boundary')
			return r, done

		######### pressure with other cells control rule ################
		######### (1) dist>0.6*r, ok  (2)0.3*r<dist<0.6*r, bad    (3)dist<0.3*r, dead
		for item in self.state_value_dict.keys():
			# print(r)
			if item != self.ai_cell:
				cell_location = np.copy(self.state_value_dict[item])
				cell_location[2] = cell_location[2] * self.plane_resolution
				dist = np.linalg.norm(cell_location - ai_location)
				cell_radius = self.get_radius(item)
				sum_radius = cell_radius + ai_radius
				dead_factor = 0.4
				ok_factor = 0.7
				if dist > ok_factor * sum_radius:
					r += 0
				elif dist > dead_factor * sum_radius and dist <= ok_factor * sum_radius:
					r += (ok_factor - float(dist) / sum_radius) / (dead_factor - ok_factor)		## 0.4->-1, 0.6->0
					
				elif dist < dead_factor * sum_radius:
					print('hit other cell:', item)
					r = -10
					done = True
					return r, done

		return r, done



	def get_ai_next_location_offset(self,a):
		offset_45 = self.ai_cell_speed * pow(2,0.5) / 2.0
		if a == 0:
			offset = np.array((0, - self.ai_cell_speed, 0))
		elif a == 1:
			offset = np.array((0, self.ai_cell_speed, 0))
		elif a == 2:
			offset = np.array((- self.ai_cell_speed, 0, 0))
		elif a == 3:
			offset = np.array((self.ai_cell_speed, 0, 0))

		elif a == 4:
			offset = np.array((- offset_45, -offset_45, 0))
		elif a == 5:
			offset = np.array((offset_45, -offset_45, 0))
		elif a == 6:
			offset = np.array((- offset_45, offset_45, 0))
		elif a == 7:
			offset = np.array((offset_45, offset_45, 0))
		elif a == 8:
			offset = np.array((0, 0, 0))
		return offset

	def step(self, a):
		done = False
		if self.ticks > 0 and self.ticks % self.tick_resolution == 0:
			self.update_stage_destination()
			if SAVE_FIG:
				name = 'screenshots_' + str(self.stage_destination_point-1) + '.ps'
				self.canvas.postscript(file=name, colormode='color')

		self.set_cell_next_location(ai_action=a)
		self.schedule.step()
		self.ticks += 1

		s_ = self.get_state()
		r, done = self.get_reward()
		if self.ticks == self.end_tick:
			done = True
		return s_, r, done



class CellAgent(Agent):
	def __init__(self, unique_id, model, name, location, diameter):
		super().__init__(unique_id, model)
		self.cell_name = name
		self.location = location
		self.diameter = diameter
		self.min_dist_list = []
		self.next_location = None

		self.set_state()

	def set_state(self):
		if self.cell_name == self.model.ai_cell:
			self.model.state_value_dict[self.cell_name] = self.location
		elif self.cell_name in self.model.state_cell_list:
			self.model.state_value_dict[self.cell_name] = self.location		

	def move(self):
		self.location = self.next_location
		self.next_location = None

	def step(self):
		self.move()
		self.set_state()


if __name__ == '__main__':
	model = SeqRosModel()
	for i_episode in range(10):
		s = model.reset()
		counter = 0
		r_overall = 0
		while True:
			model.render()
			a = 2
			s_, r, done = model.step(a)
			counter += 1
			r_overall += r
			if done:
				break
		print('Episode:', i_episode, 'Done in', counter, 'steps. Reward:',r_overall)

