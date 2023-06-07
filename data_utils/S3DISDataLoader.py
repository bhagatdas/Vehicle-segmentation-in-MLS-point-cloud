import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        #room will contain the file inside the data/mnnit/ dir in sorted order
        rooms = sorted(os.listdir(data_root))
        #store only the file name start with Area_ ['Area_1']
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            # if test_area is 5 then remaining area will consider as train_set and put the Train_set data in room_split
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            # put test_area in rooms_split 
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(2)
        #print(labelweights)
        #print(rooms_split) ['Area_2_section_2.npy', 'Area_3_section_3.npy']

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            #print(room_path)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            #print(room_data[0:2, : ])
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(3))
            #print(np.histogram(labels))
            #print(tmp,_) [123836   3238 118473  49981      0]
            #print(labelweights) adding all xyzrgl in this [273350.   5516. 242572. 110706.      0.]
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            # storing all points1, points2    and   label1, label2
            self.room_points.append(points), self.room_labels.append(labels)
            #print(self.room_coord_min)) [array([0., 0., 0.]), array([0., 0., 0.])] similar for other
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            #print(num_point_all) [295528, 336616] 
        
        labelweights = labelweights.astype(np.float32)
        #print(labelweights) [273350.   5516. 242572. 110706.      0.]
        #Normalizaing the weights
        labelweights = labelweights / np.sum(labelweights)
        # if getting inf or NaN means there is no value in one of the classes (here cluter)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        #normalizing the lables ( or y )
        sample_prob = num_point_all / np.sum(num_point_all)
        # calculating number of inter in each epoch
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        # print(num_iter) 
        room_idxs = []
        # rooms_split is the number of test or train case
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        #print(self.room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]
        #print(points[:5, :])

        while (True):
            center = points[np.random.choice(N_points)][:3]
            #print(center)
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            #print("min",block_min)
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            #print("max",block_max)
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #print(point_idxs.size)
            if point_idxs.size > 64:
                break

        if point_idxs.size >= self.num_point:
            #print("if")
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
            #print(selected_point_idxs)
        else:
            #print("else")
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        #print(self.block_points) ->. input 200 here
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
            #print(self.file_list)   ['Area_1_section_1.npy']
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            #print(root + file)  data/stanford_indoor3d/Area_1_section_1.npy
            data = np.load(root + file)
            #print(data[:3, :])   [[175.28  82.76   2.51   0.   251.    60.     4.  ]
            points = data[:, :3]
            #print(points.shape)
            self.scene_points_list.append(data[:, :6])    # whole points
            
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)
        #print(len(self.semantic_labels_list[0])) #-> 301047 (N-1*1)

        labelweights = np.zeros(2)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(3))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        #print(points.shape)  (301047, 6)
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        #print(coord_min, coord_max) [0. 0. 0.] [176.19 111.26  21.22]
        #print(self.block_size) 1.0
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        print(grid_x) #352
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        print(grid_y)  #222
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            print(index_y)
            for index_x in range(0, grid_x): # row wize data axis
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
             
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                #print(point_idxs.size)
                num_batch = int(np.ceil(point_idxs.size / self.block_points))  #getting 1
                
                point_size = int(num_batch * self.block_points)
                #print(point_size)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                #print(point_idxs_repeat.shape) (142,)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                #print(point_idxs.shape)  (200,)
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                #print(data_batch.shape) (200, 9)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
                #print("H")
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        print(data_room.shape)
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/content/drive/MyDrive/toronto/data/output/'
    num_point, test_area, block_size, sample_rate = 200, 1, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()