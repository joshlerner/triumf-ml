import numpy as np
import uproot as ur
import pickle
import awkward as ak

import os
import time

from multiprocessing import Process, Queue, set_start_method

import tensorflow as tf

import matplotlib.pyplot as plt

class Generator:
    """ """
    def __init__(self, generator_name, save_type, file_list, cellGeo_file, batch_size, 
                 labeled=True, shuffle=True, num_procs=32, preprocess=False, output_dir=None):
        """ Initialization """
        self.name = generator_name
        self.save_type = save_type
        self.labeled = labeled
        self.preprocess = preprocess
        self.output_dir = output_dir
        self.cellGeo_file = cellGeo_file
        self.cellGeo_data = ur.open(self.cellGeo_file)['CellGeo']
        # check this with original convert script
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.file_list = file_list
        self.num_files = len(self.file_list)
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.num_procs = np.min([num_procs, self.num_files])
        self.procs = []
        
        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

    def preprocessor(self, worker_id):
        """ Abstract Method """
        pass
    
    def preprocessed_worker(self, worker_id, batch_queue):
        """ """
        batch_data = []
        batch_targts = []
        
        file_num = worker_id
        while file_num < self.num_files:
            with gzip.open(self.output_dir + f'{self.name}_{file_num:03d}.{self.save_type}') as f:
                file_data = pickle.load(f)
            
            for i in range(len(file_data)):
                batch_data.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
            
                if len(batch_data) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1, 2]).astype(np.float32)
                    
                    batch_queue.put((batch_data, batch_targets))
                    
                    batch_graphs = []
                    batch_targets = []
                    
            file_num += self.num_procs
        
        if len(batch_data) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
            batch_queue.put((batch_data, batch_targets))
    
    def preprocess_data(self):
        """ """
        print('\nPreprocessing and saving data to {}'.format(self.output_dir))
        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
            
        for p in self.procs:
            p.join()
    
    def worker(self, worker_id, batch_queue):
        """ """
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('\nPreprocessing is required for combined classification/regression models')
    
    def check_procs(self):
        """ """
        for p in self.procs:
            if p.is_alive(): return True
        return False
        
    def kill_procs(self):
        """ """
        for p in self.procs:
            p.kill()
            
        self.procs = []
    
    def generator(self): ## can call fit with a generator
        """ Generator that returns processed batches during training """
        
        batch_queue = Queue(2 * self.num_procs)
        
        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)
            
        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue
                
            yield batch
            
        for p in self.procs:
            p.join()
            
class garnetDataGenerator(Generator):
    """ """
    def __init__(self, file_list, cellGeo_file, batch_size,
                labeled=True, shuffle=True, num_procs=32, preprocess=False, output_dir=None):
        """ """
        super().__init__('garnet', 'p', file_list, cellGeo_file, batch_size,
                         labeled, shuffle, num_procs, preprocess, output_dir)
        
    def preprocessor(self, worker_id):
        """ """
        file_num = worker_id
        while file_num < self.num_files:
            print(f'\nProccessing file {file_num}')
            if self.labeled:
                file = ur.open(self.file_list[file_num][0])
                label = self.file_list[file_num][1]
            else:
                file = ur.open(self.file_list[file_num])
            tree = file['EventTree']
            
            preprocessed_data = []
            
            num_events = len(tree.arrays(tree.keys()[0]))
            
            for event in range(num_events):
                
                num_clusters = tree.arrays('nCluster')['nCluster'][event]
                for cluster in range(num_clusters):
                    
                    cluster_E = tree.arrays('cluster_E')['cluster_E'][event][cluster]
                    target_E = tree.arrays('cluster_ENG_CALIB_TOT')['cluster_ENG_CALIB_TOT'][event][cluster]
                    
                    cluster_eta = tree.arrays('cluster_Eta')['cluster_Eta'][event][cluster]
                    cluster_phi = tree.arrays('cluster_Phi')['cluster_Phi'][event][cluster]
                        
                    cell_e = tree.arrays('cluster_cell_E')['cluster_cell_E'][event][cluster]
                        
                    cellIDs = tree.arrays('cluster_cell_ID')['cluster_cell_ID'][event][cluster]
                    ### map to coordinates
                    cell_eta = 
                    cell_phi =
                    cell_samp =
                    ### normalize
                    cell_eta = cell_eta - cluster_eta
                    cell_phi = cell_phi - cluster_phi
                    cell_e = np.nan_to_num(np.log10(cell_e), neginf=0.0)
                    target_E = np.nan_to_num(np.log10(target_E), neginf=0.0)
                    ### clip and pad to PADLENGTH
                    PADLENGTH = 128
                    data = ak.pad_none(np.stack((cell_eta, cell_phi, cell_samp, cell_e), axis=-1), PADLENGTH, clip=True, axis=0)
                    if not self.labeled:
                        label = np.round(tree.arrays('cluster_EM_PROBABILITY')['cluster_EM_PROBABILITY'][event][cluster]
                    target = [label, int(not label), target_E]
                    if cluster_E > 0:
                        preprocessed_data.append((data, target))
            
            with gzip.open(self.output_dir + f'{self.name}_{file_num:03d}.{self.save_type}', 'wb') as f:
                pickle.dump(preprocessed_data, f)
            
            print(f'\nFinished processing file {file_num}')
            file_num += self.num_procs
        
    
        