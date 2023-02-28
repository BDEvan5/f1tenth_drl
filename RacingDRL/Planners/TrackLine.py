import csv
import numpy as np
import  matplotlib.pyplot as plt
from numba import njit

class TrackLine:
    def __init__(self, map_name, racing_line=False, expand=False) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = map_name
        self.total_s = None
        self.vs = None
        
        if racing_line:
            self.load_raceline()
        else:
            self.load_centerline()
            
        if expand:
            self._expand_wpts()
            
            
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        self.max_distance = 0
        self.distance_allowance = 1

        self.load_centerline()

    def load_centerline(self):
        filename = 'maps/' + self.map_name + '_centerline.csv'
        xs, ys, w_rs, w_ls = [0], [0], [], []
        with open(filename, 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))
                w_rs.append(float(lines[2]))
                w_ls.append(float(lines[3]))
        xs[-1] = 0
        ys[-1] = 0
        self.xs = np.array(xs)[:, None]
        self.ys = np.array(ys)[:, None]
        self.centre_length = len(xs)

        self.wpts = np.vstack((xs, ys)).T

        diffs = np.diff(self.wpts, axis=0)
        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)

        self.total_s = self.ss[-1]
        self.N = len(self.wpts)
    
    def load_raceline(self):
        track = []
        filename = 'maps/' + self.map_name + "_raceline.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.wpts = track[:, 1:3]
        self.vs = track[:, 5] 

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        self.total_s = self.ss[-1]
    
    def _expand_wpts(self):
        n = 5 # number of pts per orig pt 
        dz = 1 / n
        o_line = self.wpts
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.wpts = np.array(new_line)
        self.vs = np.array(new_vs)


    def plot_wpts(self):
        plt.figure(1)
        plt.plot(self.wpts[:, 0], self.wpts[:, 1], 'b-')
        for i, pt in enumerate(self.wpts):
            plt.text(pt[0], pt[1], f"{i}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    def get_lookahead_point(self, position, lookahead_distance):
        wpts = np.vstack((self.wpts[:, 0], self.wpts[:, 1])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts, self.l2s, self.diffs)

        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
        if i2 == None: 
            return None
        lookahead_point = np.empty((3, ))
        lookahead_point[0:2] = wpts[i2, :]
        lookahead_point[2] = self.vs[i]
        
        return lookahead_point

    def calculate_progress_percent(self, point):
        idx, dists = self.get_trackline_segment(point)

        x, h = self.interp_pts(idx, dists)

        s = self.ss[idx] + x
        s_percent = s/self.total_s
        
        return s_percent

    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.ss[idx+1] - self.ss[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            Area = Area_square**0.5
            h = Area * 2/d_ss
            if np.isnan(h):
                h = 0
            x = (d1**2 - h**2)**0.5

        return x, h

    def get_trackline_segment(self, point):
        """Returns the first index representing the line segment that is closest to the point.

        wpt1 = pts[idx]
        wpt2 = pts[idx+1]

        dists: the distance from the point to each of the wpts.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists

    def get_cross_track_heading(self, point):
        idx, dists = self.get_trackline_segment(point)
        point_diff = self.wpts[idx+1, :] - self.wpts[idx, :]
        trackline_heading = np.arctan2(point_diff[1], point_diff[0])

        x, h = self.interp_pts(idx, dists)

        return trackline_heading, h

    def plot_vehicle(self, point, theta):
        idx, dists = self.get_trackline_segment(point)
        point_diff = self.wpts[idx+1, :] - self.wpts[idx, :]
        trackline_heading = np.arctan2(point_diff[1], point_diff[0])

        x, h = self.interp_pts(idx, dists)

        track_pt = self.wpts[idx] + x * np.array([np.cos(trackline_heading), np.sin(trackline_heading)])

        plt.figure(1)
        plt.clf()
        size = 1.2
        plt.xlim([point[0]-size, point[0]+size])
        plt.ylim([point[1]-size, point[1]+size])
        plt.plot(self.wpts[:,0], self.wpts[:,1], 'b-x', linewidth=2)
        plt.plot(self.wpts[idx:idx+2, 0], self.wpts[idx:idx+2, 1], 'r-', linewidth=2)
        plt.plot([point[0], track_pt[0]], [point[1], track_pt[1]], 'orange', linewidth=2)
        plt.plot(track_pt[0], track_pt[1],'o', color='orange', markersize=6)

        plt.plot(point[0], point[1], 'go', markersize=6)
        plt.arrow(point[0], point[1], np.cos(theta), np.sin(theta), color='g', head_width=0.1, head_length=0.1, linewidth=2)

        plt.pause(0.0001)

    def check_done(self, observation):
        position = observation['state'][0:2]
        s = self.calculate_progress(position)

        if s <= (self.max_distance - self.distance_allowance) and self.max_distance < 0.8*self.total_s and s > 0.1:
            # check if I went backwards, unless the max distance is almost finished and that it isn't starting
            return True # made negative progress
        self.max_distance = max(self.max_distance, s)

        return False


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory_py2(point, trajectory, l2s, diffs):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=True, cache=True)
def add_locations(x1, x2, dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret

@njit(fastmath=True, cache=True)
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret
