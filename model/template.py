import pymesh
import numpy as np
import torch
from torch.autograd import Variable
from random import randrange, random
from math import sqrt
import math

"""
        Author : Thibault Groueix 01.11.2019
"""


def get_template(template_type, device=0):
    getter = {
        "SQUARE": SquareTemplate,
        "SPHERE": SphereTemplate,
        "HEXAGON": HexagonTemplate,
    }
    template = getter.get(template_type, "Invalid template")
    return template(device=device)


class Template(object):
    def get_random_points(self):
        print("Please implement get_random_points ")

    def get_regular_points(self):
        print("Please implement get_regular_points ")


class HexagonTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    @staticmethod
    def randinunithex():
        vectors = [(-1.,0),(.5,sqrt(3.)/2.),(.5,-sqrt(3.)/2.)]
        i = randrange(3)
        v1, v2 = vectors[i], vectors[(i+1)%3]
        x, y = random(),random()
        return torch.tensor([x*v1[0]+y*v2[0],x*v1[1]+y*v2[1]])
    
    @staticmethod
    def polygon(sides, radius=1, rotation=0, translation=None):
        one_segment = math.pi * 2 / sides

        points = [
            (math.sin(one_segment * i + rotation) * radius,
             math.cos(one_segment * i + rotation) * radius)
            for i in range(sides)]

        if translation:
            points = [[sum(pair) for pair in zip(point, translation)]
                      for point in points]

        return points
    
    @staticmethod
    def calculate_polygons(startx, starty, endx, endy, radius):
        """ 
        Calculate a grid of hexagon coordinates of the given radius
        given lower-left and upper-right coordinates 
        Returns a list of lists containing 6 tuples of x, y point coordinates
        These can be used to construct valid regular hexagonal polygons

        You will probably want to use projected coordinates for this
        """
        # calculate side length given radius   
        sl = (2 * radius) * math.tan(math.pi / 6)
        # calculate radius for a given side-length
        # (a * (math.cos(math.pi / 6) / math.sin(math.pi / 6)) / 2)
        # see http://www.calculatorsoup.com/calculators/geometry-plane/polygon.php

        # calculate coordinates of the hexagon points
        # sin(30)
        p = sl * 0.5
        b = sl * math.cos(math.radians(30))
        w = b * 2
        h = 2 * sl

        # offset start and end coordinates by hex widths and heights to guarantee coverage     
        startx = startx - w
        starty = starty - h
        endx = endx + w
        endy = endy + h

        origx = startx
        origy = starty


        # offsets for moving along and up rows
        xoffset = b
        yoffset = 3 * p

        polygons = []
        row = 1
        counter = 0

        while starty < endy:
            if row % 2 == 0:
                startx = origx + xoffset
            else:
                startx = origx
            while startx < endx:
                p1x = startx
                p1y = starty + p
                p2x = startx
                p2y = starty + (3 * p)
                p3x = startx + b
                p3y = starty + h
                p4x = startx + w
                p4y = starty + (3 * p)
                p5x = startx + w
                p5y = starty + p
                p6x = startx + b
                p6y = starty
                poly = [
                    [p1x, p1y],
                    [p2x, p2y],
                    [p3x, p3y],
                    [p4x, p4y],
                    [p5x, p5y],
                    [p6x, p6y],
                    [(p2x + p5x)/2,(p2y + p5y)/2]]
                if -1 < p1x < 1 and -1 < p1y < 1 and -1 < p2x < 1 and -1 < p2y < 1 and -1 < p3x < 1 and -1 < p3y < 1 and -1 < p4x < 1 and -1 < p4y < 1 and -1 < p5x < 1 and -1 < p5y < 1 and -1 < p6x < 1 and -1 < p6y < 1:
                    polygons.append(poly)
#                 else:
#                     print(poly)
                counter += 1
                startx += w
            starty += yoffset
            row += 1
        #print(len(polygons))            
        return polygons
    
    def generate_hexagon_old(self, grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        #grain = int(grain)
        #grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        vertices=self.polygon(6)
        vertices.append((0,0))
        for x in range(6):
            #print(b[x][0],b[x][1])
            a1=(x+1)%6
            a2=(x-1)%6
            faces.append([x,a1,6])
            faces.append([x,a2,6])
        
        return np.array(vertices), np.array(faces)
    
    def generate_hexagon(self, grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        #grain = int(grain)
        #grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        radius=0.15
        a = self.calculate_polygons(0,0,1,1,radius)
        for x in a:
            for y in x:
                vertices.append(y)
        for x in range(len(a)):
            for y in range(6):
                faces.append([7*x+y,7*x+((y+1)%6),7*(x+1)-1])
        print(len(vertices), len(a))
        return np.array(vertices), np.array(faces)    
    
    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Hexagon
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        for n in range(shape[-1]):
            rand_grid[0,:,n] = self.randinunithex()
        assert rand_grid.shape == shape
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        """
        Get regular points on a Hexagon
        Return Tensor of Size [x, 2]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_hexagon(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)  # 10k vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)

        return Variable(self.vertex[:, :2].contiguous().to(device))
    
class SphereTemplate(Template):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, "shape should have 3 in dim 1"
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None, device="gpu0"):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.mesh = pymesh.generate_icosphere(1, [0, 0, 0], 4)  # 2562 vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)
            self.npoints = npoints

        return Variable(self.vertex.to(device))


class SquareTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        #print(rand_grid)
        #print(rand_grid.shape)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        """
        Get regular points on a Square
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)  # 10k vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)

        return Variable(self.vertex[:, :2].contiguous().to(device))

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)
