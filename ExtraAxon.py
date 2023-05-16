from neuron import h
import numpy as np
h.load_file("stdrun.hoc")
    
class Axon:
    name = "axon"

    def __init__(self, gid , diam, L, mode = 'passive', nseg_interv = 40):
        self._gid = gid 
        self.mode = mode       
        self._setup_morphology(diam,L,nseg_interv)        
        self.all = self.axon.wholetree()
        self._setup_biophysics()                
        h.define_shape()            
        self._setpointers() # set up xtra mechanism for extracellular stimulation        
        self._setup_recordings()        
        
    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)

    def _setup_morphology(self,diam,L,nseg_interv=40):
        self.axon = h.Section(name='axon',cell=self)        
        self.axon.L = L
        self.axon.diam = diam
        self.axon.nseg = 1 + 2*int(L/nseg_interv)        

    def _setup_biophysics(self):
        for sec in self.all:            
            sec.insert('extracellular')
            sec.insert('xtra')
            sec.Ra = 100
            sec.cm = 1        
        if self.mode == 'passive':
            for sec in self.all:
                sec.insert('pas')
                sec.g_pas = 5e-4 # S/cm2
                sec.e_pas = 0 # mV
        elif self.mode == 'hh':
            for sec in self.all:
                sec.insert('hh')

    def _setpointers(self):
        self._getcoords()
        for sec in self.all:
             for seg in sec:
                h.setpointer(sec(seg.x)._ref_e_extracellular, 'ex', sec(seg.x).xtra)


    def _setup_recordings(self):
        self._spike_detector = h.NetCon(self.axon(0.5)._ref_v,None,sec=self.axon)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times) # Record spike times to spike_times Vector        
        self.seg_vs = [h.Vector().record(seg._ref_v) for seg in self.axon] # Record voltage in each segment        
        self.x_vals = np.array([seg.x for seg in self.axon]) # normalized position (x) along axon for each segment, ranges from 0 to 1
    
    def _getcoords(self):
        # get coordinates of segment centers
        xcoords = []
        ycoords = []
        zcoords = []

        for sec in self.all: 
            nn = sec.n3d() # number of pt3d points
            xx = h.Vector(nn) # create xx vector length of nn
            yy = h.Vector(nn)
            zz = h.Vector(nn)
            length = h.Vector(nn)
            for ii in range(0,nn):
                xx[ii] = sec.x3d(ii)
                yy[ii] = sec.y3d(ii)
                zz[ii] = sec.z3d(ii)
                length[ii] = sec.arc3d(ii)
            
            # normalize length along centroid
            length = length/(length[-1])

            # initialize new veector for interpolated values

            range_vec = h.Vector(sec.nseg + 2) # nseg + 2 because counts centers of each seg + 0 and 1
            range_vec.indgen(1/sec.nseg) # step size is 1/nseg, normalized to go from 0 to 1
            range_vec  = range_vec - 1/(2*sec.nseg)
            range_vec[0] = 0
            range_vec[sec.nseg+1] = 1

            # length contains normalized distances of pt3d points (irregular intervals)
            # range_vec contains normalized distances of center of each segment (regular intervals)
            # interpolate range_vec within length
            
            xint = np.interp(range_vec,length,xx)
            yint = np.interp(range_vec,length,yy)
            zint = np.interp(range_vec,length,zz)
            # Remove 0 and 1 end points
            xint = xint[1:-1]
            yint = yint[1:-1]
            zint = zint[1:-1]

            # assign coordinates of each segment to x_xtra, y_xtra, and z_xtra

            for ii, seg in enumerate(sec):
                seg.x_xtra = xint[ii]
                seg.y_xtra = yint[ii]
                seg.z_xtra = zint[ii]

            xcoords.append(xint)
            ycoords.append(yint)
            zcoords.append(zint)
        
        xcoords = np.concatenate(xcoords).reshape((-1,1))
        ycoords = np.concatenate(ycoords).reshape((-1,1))
        zcoords = np.concatenate(zcoords).reshape((-1,1))

        self.coords = np.concatenate([xcoords,ycoords,zcoords],axis=1)
        


