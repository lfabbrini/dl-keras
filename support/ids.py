"""
Created on Thu Mar  8 09:53:33 2018

@author: lfabbrini
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #do not shwo issue with pre-build binaries
import re
import warnings
from keras import backend as K #img_to_array
from keras.preprocessing.image import img_to_array #img_to_array
from keras import models#extract_features_from_net
from keras.utils import Sequence #HDF5DataGenerator
import h5py #HDF5DataGenerator
import sys
from time import time
from scipy.misc import imresize #imresize
from IPython.core.debugger import set_trace #debugger

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

def clip(x,x_min=None,x_max=None):
    """clip value to [x_min,x_max]
    
    # Arguments
        x: numpy array
        x_min,x_max: min/max output

    # Returns
        x: modified numpy array
    """ 
    if x_min:
        x[x<x_min] = x_min
    if x_max:
        x[x>x_max] = x_max
        
#    print(x.dtype)    
    return x

def linearmap(x,x_min=None,x_max=None,y_min=0,y_max=255):
    """linear map [x_min,x_max] in [y_min,y_max]
    
    # Arguments
        x: numpy array
        x_min,x_max: min/max input
        y_min,y_max: min/max output

    # Returns
        x: modified numpy array
    """ 
    if not x_min:
        x_min = np.min(x)
    if not x_max:
        x_max = np.max(x)    
    x = (y_max-y_min)*(x -x_min)/(x_max-x_min) + y_min
    
#    print(x.dtype)  
    return x

def resize(x,hsize=32,wsize=32):
    """linear map [x_min,x_max] in [y_min,y_max]
    
    # Arguments
        x: numpy array of size (num_images,H,W,CH)
        HSize,WSize: H/W dimension of the resized output

    # Returns
        x_r: resized numpy array
    """ 
    xmin, xmax = np.min(x), np.max(x)
    N = x.shape[0]
    C = x.shape[-1]
    x_r = np.zeros((N,hsize,wsize,C),dtype=K.floatx()) 
    for i in range(N):
        for c in range(C):
            x_r[i,:,:,c] = imresize(x[i,:,:,c],(hsize,wsize))#should have 1 or 3 channel, convert to PIL image and go back
            
    #imresize map [xmin,xmax] in [0,255] before resizing, so we have to put the limit back
    x_r = linearmap(x_r,0,255,xmin,xmax)
    
#    print(x.dtype)         
    return x_r

def get_slice(axis_size,mid_size_slice,downsampling):
    """Compute the slicing vector at the center of the axis of length axis_size with downsampling
    
    # Arguments
        i0: central index
        mid_size: semi-dimension of the interval
        downsampling: downsampling factor

    # Returns
        slice_vect: slicing vector to apply
    """ 
    samples = np.arange(1,mid_size_slice+1)*downsampling
    i0 = np.array([int(np.floor(axis_size/2.))])
    left = i0-samples
    right = i0+samples
    slice_vect = np.concatenate([left[::-1] ,i0, right])
    areOk = np.logical_and(slice_vect>= 0 , slice_vect < axis_size)
    return slice_vect[areOk]

def get_optimum_batch(x,batch_size_interval=(16,32)):
    """Compute the batchsize giving the minimum rest
    
    # Arguments
        x: numpy array of size (num_images,...)
        batch_size_interval: interval of batchsize to try

    # Returns
        (batch_opt,iter_opt,rest_opt)
        batch_opt: optimum batch size
        iter_opt: optimum number of iterations
        rest_opt: minimum number of image rest
    """ 
    NumIm = x.shape[0]
    batch_size_to_test = np.arange(batch_size_interval[0],batch_size_interval[1])
    rest = NumIm % batch_size_to_test
    rest_opt = rest.min()
    argmin = rest.argmin()
    batch_opt = batch_size_to_test[argmin]
    iter_opt = NumIm/batch_opt
    return (batch_opt,iter_opt,rest_opt)
    
def extract_features_from_sequential_model(x,model):
    """Stack layer up to flatten (exclused) and than compute output value
    
    # Arguments
        x: numpy array of size (num_images,...)
        model: net model where to call model.predict(x) (usually a pretrained model without top dense classifier)

    # Returns
        features_stack: numpy array of size (num_images,...)
    """
    is_flatten_re=re.compile(r"flatten")
    model_base = model.Sequential()
    names = [layer.name for layer in model.layers]
    for layer,name in zip(model.layers,names):
        is_flatten_res = is_flatten_re.search(name)
        if is_flatten_res != None:
            break
        model_base.add(layer)
    
    features_stack = model_base.predict(x)
    return features_stack
    
def create_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale=False,compute_mean=False):
    """Create input to load in Keras from a list of images in filename.
       In particular it writes in a compressed file .npz a numpy array of the stacked images (and their mean, if set)

    # Arguments
        filename: file with image name and label in each row (e.g. B_FA322_2017_06_30_002Swath1x0_ch0_y0.png 1) 
        path_data: path where dataset_dir are located
        dataset_dir: folder name of the dataset
        model_dir: folder name of the model
        grayscale: True if the image are gray (1-channel depth), False for RGB images
        compute_mean: True to compute the mean of oll the images listed in filename

    # Returns
        
    """
#    train_file=scan_type+'_sublist_train.txt'
#    test_file=scan_type+'_sublist_test.txt'
#    val_file=scan_type+'_sublist_val.txt'
    
#    filelist = [train_file,test_file,val_file]
#    for filename in filelist:
    #fileToOpen = path_data + '/' + model_dir + '/' + filename
    whitespace = re.compile(r"\s")    
    fileToOpen = os.path.join(path_data,dataset_dir,model_dir,filename)
    with open(fileToOpen) as f:
        imlist = [line.rstrip() for line in f]#'\n' is stripped
    M = len(imlist)
#    fileToOpenD = os.path.join(path_data,dataset_dir,model_dir,'debug.txt')
#    with open(fileToOpenD,'w') as f:
#        f.writelines( line +'\n' for line in imlist)
    if M==0:
        raise ImportError('Could not import '+fileToOpen)
    
    #take the first image in order to create np.array of the right size
    fileimage , label = whitespace.split(imlist[0])
    fileToOpen = os.path.join(path_data,dataset_dir,'img',fileimage)
    img = load_img(fileToOpen,grayscale,print_warning=True)
    ##set_trace()
    img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}
    ##set_trace()
    labels = np.zeros((M,1),dtype='uint8')
    images = np.zeros((M,)+img.shape,dtype=img.dtype)
    mu_img = np.zeros(img.shape,dtype=img.dtype)
    # setup toolbar
    #toolbar_width  = 10
    #sys.stdout.write("[%s]" % (" " * toolbar_width))
    #sys.stdout.flush()
    #read all images
    for i,line in enumerate(imlist):
        fileimage , label = whitespace.split(line)
        label = int(label)
        labels[i] = label
        fileToOpen = os.path.join(path_data,dataset_dir,'img',fileimage)
        img = load_img(fileToOpen,grayscale,print_warning=False)
        img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}
        images[i]=img
        if compute_mean:
            mu_img+=img
            if i%(M/10) == 0:
                #sys.stdout.write("\b" * (toolbar_width+1-i)) # return to start of line, after '['
                sys.stdout.write("-")
                sys.stdout.flush()
    sys.stdout.write("\n")
    #cwd = os.getcwd()    #directory where the script is called by terminal    
    cwd =  os.path.join(path_data,dataset_dir,model_dir)
    if compute_mean:        
        mu_img/=M
        filename_no_ext = filename.split('.')[0] + '_mean'
        fileToSave = os.path.join(cwd,filename_no_ext)
        np.savez_compressed(fileToSave,x=mu_img,y=-1)
        
    filename_no_ext = filename.split('.')[0]
    fileToSave = os.path.join(cwd,filename_no_ext)
    np.savez_compressed(fileToSave,x=images,y=labels)
    
def get_central_data_from_filelist(filename,path_data,dataset_dir,model_dir,grayscale=False):
    """Return the central images from a list of images in filename.
       
    # Arguments
        filename: file with image name and label in each row (e.g. B_FA322_2017_06_30_002Swath1x0_ch0_y0.png 1) 
        path_data: path where dataset_dir are located
        dataset_dir: folder name of the dataset
        model_dir: folder name of the model
        grayscale: True if the image are gray (1-channel depth), False for RGB images
    # Returns
        (x_center,y_center)(anomaly_id,acq_id)
    """
    whitespace = re.compile(r"\s")    
    anomaly_id = re.compile(r"\_(T|FA)\d+\_")
    acq_id = re.compile(r"_\d\d\d\d\_\d\d_\d\d[^x]*")
    is_central = re.compile(r"x0\_ch0\_y0\.")
    fileToOpen = os.path.join(path_data,dataset_dir,model_dir,filename)
    with open(fileToOpen) as f:
        imlist = [line.rstrip() for line in f]#'\n' is stripped
    M = len(imlist)
#    fileToOpenD = os.path.join(path_data,dataset_dir,model_dir,'debug.txt')
#    with open(fileToOpenD,'w') as f:
#        f.writelines( line +'\n' for line in imlist)
    if M==0:
        raise ImportError('Could not import '+fileToOpen)
    
    #take the first image in order to create np.array of the right size
    fileimage , label = whitespace.split(imlist[0])
    fileToOpen = os.path.join(path_data,dataset_dir,'img',fileimage)
    img = load_img(fileToOpen,grayscale,print_warning=True)
    ##set_trace()
    img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}
    ##set_trace()
    are_central = np.zeros(M,dtype='bool')
    labels = np.zeros((M,1),dtype='uint8')
    images = np.zeros((M,)+img.shape,dtype=img.dtype)
    acq_id_str = []
    anomaly_id_str=[]
    # setup toolbar
    #toolbar_width  = 10
    #sys.stdout.write("[%s]" % (" " * toolbar_width))
    #sys.stdout.flush()
    #read all images
    for i,line in enumerate(imlist):
        fileimage , label = whitespace.split(line)
        label = int(label)
        labels[i] = label
        fileToOpen = os.path.join(path_data,dataset_dir,'img',fileimage)
        is_central_res = is_central.search(fileimage)
        if is_central_res == None:
            continue
        are_central[i]=True
        anomaly_id_res = anomaly_id.search(fileimage)
        acq_id_res = acq_id.search(fileimage)
        if acq_id_res.group() != None:
            str_out = acq_id_res.group()
            acq_id_str.append(str_out[1:])
        else:
            acq_id_str.append('-')
        if anomaly_id_res.group() != None:
            str_out = anomaly_id_res.group()
            anomaly_id_str.append(str_out[1:-1])
        else:
            anomaly_id_str.append('-')    
        
        img = load_img(fileToOpen,grayscale,print_warning=False)
        img = img_to_array(img) #automatic detect Keras backend data_format in {'channels_first', 'channels_last'}
        images[i]=img
    ##set_trace()
    images = images[are_central]
    labels = labels[are_central]
    return (images,labels),(anomaly_id_str,acq_id_str)
 
def load_img(fileToOpen,grayscale=False,print_warning=False):
    """Load a image with PIL

    # Arguments
        fileToOpen: Image location.
        grayscale: Image color depth.
        print_warning: Enable warning.

    # Returns
        A PIL Image Object.

    # Raises
        ImportError: if Could not import PIL.Image.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                      'The use of `array_to_img` requires PIL.')
    img = pil_image.open(fileToOpen)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
            if print_warning:
                warnings.warn('Source image are converted to grayscale', Warning)
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            if print_warning:
                warnings.warn('Source image are converted to RGB', Warning)
                
    return img

def resize_img(img,target_size,interpolation='bilinear'):
    """Resize a image with PIL

    # Arguments
        img: a PIL Image Object.
        target_size: touple with the output size.

    # Returns
        A PIL Image Object.

    # Raises
        ValueError: if Could not find PIL.Image interpolation method
    """
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img
    
    
    
def get_dataset_id(dataset_dir):
    """Get Dataset String to identify it

    # Arguments
        dataset_dir: dataset folder.

    # Returns
        id_str: ID string.

    """
    num_acq_re = re.compile(r"NAcq\d+")
    pfa_re = re.compile(r"P\de\d+")
    dataset_id_re = re.compile(r"Tex\d+\_\d+")
    data_re = re.compile(r"[a-zA-Z]+$")
    
    id_str =''
    list_re = []
    list_re.append(num_acq_re)
    list_re.append(pfa_re)
    list_re.append(dataset_id_re)
    list_re.append(data_re)
    for rr in list_re:
        rr_res=rr.search(dataset_dir)
        if rr_res != None:
            id_str +=rr_res.group()+'_'
    return id_str


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model, layer_name, filter_index, input_shape):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the loss wrt the input
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    #input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    input_img_data = np.random.random(input_shape) * 2  - 1 #generate value [-1, 1]

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)



def slice_volume(x,scan_type,mid_size_slice,downsampling_xyz):
    """Elaborate input volume so to have the desired "face" (scan_type) in front of the volume and downsampling if required
       It always take the central slice of volume 
    # Arguments
        batch_size: number of data for each batch.
        file_to_open: hdf5 file with /data and /label dataset.

    """
#    #set_trace()
    #input shape (N,CH,X,Z) 
    Z = x.shape[3]
    X = x.shape[2]
    Y = x.shape[1]
    if scan_type == 'B':
        slice_vect = get_slice(Y,mid_size_slice,downsampling_xyz[1])
        #input shape (N,CH,X,Z)
        x = x[:,slice_vect,::downsampling_xyz[0],::downsampling_xyz[2]] 
        #output shape (N,Z,X,CH)
        x = x.transpose((0,3,2,1))
    elif scan_type == 'T':
        slice_vect = get_slice(X,mid_size_slice,downsampling_xyz[0])
        #input shape (N,CH,X,Z) 
        x = x[:,::downsampling_xyz[1],slice_vect,::downsampling_xyz[2]]
        #output shape (N,Z,CH,X)
        x = x.transpose((0,3,1,2))
    elif scan_type == 'C':
        slice_vect = get_slice(Z,mid_size_slice,downsampling_xyz[2])
        #input shape (N,CH,X,Z) 
        x = x[:,::downsampling_xyz[1],::downsampling_xyz[0],slice_vect]
        #output shape (N,CH,X,Z)
#    #set_trace()
    return x
    
class HDF5DataGenerator(Sequence):
    """Generate Data from a dataset saved in a hdf5file
       Inherited from Sequence (so enabling multiprocessing pre-fetching)  
    # Arguments
        batch_size: number of data for each batch.
        file_to_open: hdf5 file with /data and /label dataset.

    """
    def __init__(self,file_to_data=None,file_to_mean=None,batch_size=None,shuffle=True,seed=0,preprocessing_function_list=None):
#        #set_trace()
        if file_to_data is None:
            raise ValueError('Invalid file_to_open')
        self.file_to_data = file_to_data
        f = h5py.File(file_to_data, 'r')
        self.x, self.y = f['data'], f['label'] #x.shape is (N,H,W,CH)
        
        if batch_size is None:
            (batch_size,_,_) = get_optimum_batch(self.x)#return (batch_opt,iter_opt,rest_opt)
        self.batch_size = batch_size
        
        self.file_to_mean = file_to_mean
        if file_to_mean is not None:
            f = h5py.File(file_to_mean, 'r')
            self.x_mu = f['data'][:] #avoid accessig to the disk for the mean and load it in RAM
            
        self.shuffle = shuffle
        self.preprocessing_function_list = preprocessing_function_list
        self.n = self.y.shape[0]
#        self.batch_number = int(np.floor(self.n / self.batch_size)) #round down and rest
        
        self.batch_number = int(np.ceil(self.n / self.batch_size)) #round up and wrap
        
        self.seed = seed
        np.random.seed(seed)
        
#        #shuffle sample order      
#        self.indexes = np.arange(self.x.shape[0])
#        if self.shuffle:
#            np.random.shuffle(self.indexes)
        
        #shuffle the batch order   
        self.indexes = np.arange(self.batch_number)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        if self.file_to_mean is not None:
            if self.preprocessing_function_list:
                for i,f in enumerate(self.preprocessing_function_list):
                    self.x_mu = f(self.x_mu)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batch_number
        
    def __getitem__(self, idx):
        'Generate one batch of data'
        if self.shuffle:
            idx = self.indexes[idx]#shuffled idx

        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        if  stop > self.n-1:
            rest = stop % self.n
            batch_x = np.concatenate((self.x[slice(start,self.n)],self.x[slice(0,rest)]))
            batch_y = np.concatenate((self.y[slice(start,self.n)],self.y[slice(0,rest)]))
        else:
            batch_x = self.x[slice(start,stop)]
            batch_y = self.y[slice(start,stop)]



        if self.preprocessing_function_list:
            for i,f in enumerate(self.preprocessing_function_list):
                batch_x = f(batch_x)
            
            
        if self.file_to_mean is not None:
            batch_x -= self.x_mu

        return batch_x, batch_y    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
class ScanDataGenerator(HDF5DataGenerator):
    """Generate Data, reassembling the volume so to have the desired "face" (scan_type) in front of the volume and downsampling if required
        
    # Arguments
        scan_type: volume cut among 'B','T','C'.
        downsampling_xyz: downsampling to apply to data.

    """      
    def __init__(self,file_to_data=None,file_to_mean=None,batch_size=None,shuffle=True,preprocessing_function_list=None,scan_type='C',stacked_scan=3,downsampling_xyz=None):
        super(ScanDataGenerator,self).__init__(file_to_data=file_to_data,file_to_mean=file_to_mean,batch_size=batch_size,shuffle=shuffle,
             preprocessing_function_list=None)#preprocessing_function_list=None so that self.x_mu is only loaded
#        #set_trace()
        self.preprocessing_function_list = preprocessing_function_list
        if downsampling_xyz is None:
            self.downsampling_xyz=np.array(np.ones((1,3)))
        self.downsampling_xyz = downsampling_xyz     
        
        if scan_type not in {'B', 'T', 'C'}:
            raise ValueError('Invalid scan_type:', scan_type,
                             '; expected one in "B", "T", "C"')
        self.scan_type = scan_type
        
        mid_size_slice = int(np.floor(stacked_scan/2.))
        self.mid_size_slice = mid_size_slice

        if self.file_to_mean is not None:#apply the correct processing step
            #slice x_mu
            self.x_mu = slice_volume(self.x_mu,self.scan_type,self.mid_size_slice,self.downsampling_xyz)
            if self.preprocessing_function_list:
                for i,f in enumerate(self.preprocessing_function_list):
                    self.x_mu = f(self.x_mu)

#            print('mu',self.x_mu.shape)
    def __getitem__(self, idx):
        'Generate one batch of data'
        if self.shuffle:
            idx = self.indexes[idx]#shuffled idx
        
        start = idx * self.batch_size
        stop = (idx + 1) * self.batch_size
        if  stop > self.n-1:
            rest = stop % self.n
            batch_x = np.concatenate((self.x[slice(start,self.n)],self.x[slice(0,rest)]))
            batch_y = np.concatenate((self.y[slice(start,self.n)],self.y[slice(0,rest)]))
        else:
            batch_x = self.x[slice(start,stop)]
            batch_y = self.y[slice(start,stop)]
        
        batch_x = slice_volume(batch_x,self.scan_type,self.mid_size_slice,self.downsampling_xyz)
        
#        print('b',batch_x.shape)  
        if self.preprocessing_function_list:
            for i,f in enumerate(self.preprocessing_function_list):
                batch_x = f(batch_x)
                
#        print('b',batch_x.shape)        
        if self.file_to_mean is not None:
            batch_x -= self.x_mu

        return batch_x, batch_y    
        
    def shape(self):
        x = self.x[0:1]
        x = slice_volume(x,self.scan_type,self.mid_size_slice,self.downsampling_xyz)
        if self.preprocessing_function_list:
            for i,f in enumerate(self.preprocessing_function_list):
                x = f(x)
        return x.shape