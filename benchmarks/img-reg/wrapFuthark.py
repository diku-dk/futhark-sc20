import numpy
import pyopencl
import pyopencl.array as pycl_array

#from .imgRegHisto import imgRegHisto
import imgRegHisto

class HISTOFUTH:

    def __init__(self,
                 data1,
                 data2,
                 verbose=0,
                 platform_id=0,
                 device_id=0
                ):
        
        self.verbose = verbose
        self.platform_id = platform_id
        self.device_id = device_id
        
        # initialize device
        self._init_device(platform_id, device_id)

        self.futobj = imgRegHisto.imgRegHisto(
                                    command_queue=self.queue, 
                                    interactive=False,
                                    platform_pref=self.platform_id,
                                    device_pref=self.device_id,
                                    default_group_size=256,
                                    default_num_groups=144)

        self.data1_cl = pycl_array.to_device(self.queue, data1)
        self.data2_cl = pycl_array.to_device(self.queue, data2)
            
    def _init_device(self, platform_id, device_id):
        """ Initializes the device.
        """
                
        try:
            platforms = pyopencl.get_platforms()
            devices = platforms[platform_id].get_devices()
        except Exception as e:
            raise Exception("Could not access device '{}' on platform with '{}': {}".format(str(platform_id), str(device_id), str(e)))
        
        self.device = devices[device_id]
        self.ctx = pyopencl.Context(devices=[self.device])
        self.queue = pyopencl.CommandQueue(self.ctx)
    
    def _print_device_info(self):
        """ Prints information about the current device.
        """
        
        if self.verbose > 0:
            print("=================================================================================")
            print("Device id: " + str(self.device_id))
            print("Device name: " + str(self.device.name()))
            print("---------------------------------------------------------------------------------")
            print("Attributes:\n")
            attributes = self.device.get_attributes()
            for (key, value) in attributes.iteritems():
                print("\t%s:%s" % (str(key), str(value)))            
            print("=================================================================================")    
                    
    def computeHistos(self):
        """ compute the three histograms
        
        ----------
        Parameters
        ----------
        data1 and dat2: unidimensional arrays of shape (N),        
        -------
        Returns: three histograms
        -------
        """

        h1, h2, hist1_cl, hist2_cl, histc_cl = self.futobj.mkImgRegHisto(self.data1_cl, self.data2_cl)

        # get the numpy array with hist1_cl.get()
        return h1, h2, hist1_cl, hist2_cl, histc_cl