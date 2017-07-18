import random
import math
import numpy as np
from operator import itemgetter

class Dataset(object):
    def __init__(self,id,x,y,meta={}):
        self.id=id
        self.x=x
        self.y=y
        self.meta=meta

    def classes(self):
        return max(self.y)+1

    def remove_classes_with_few_examples(self,min_examples):
        classes=self.classes()

        histogram=np.zeros(classes,dtype=int)
        for i in range(len(self.y)):
            histogram[self.y[i]]=histogram[self.y[i]]+1
        classes_to_keep = np.where(min_examples <= histogram)[0]
        indices_to_keep = np.where(np.in1d(self.y, classes_to_keep))[0]
        if len(indices_to_keep)>0:
            getter=itemgetter(*list(indices_to_keep))
            d=dict(self.meta)
            d['x']=self.x
            d['y']=self.y
            for key in d:
                if len(d[key])==len(self.y):
                    part=d[key]
                    d[key]=list(getter(part))
            self.x=d.pop("x")
            self.y=d.pop("y")
            classes_to_keep_list=list(classes_to_keep)
            print(self.y)
            for i in range(len(self.y)):
                new_class_label=classes_to_keep_list.index(self.y[i])
                self.y[i]=new_class_label
            print(self.y)
        return indices_to_keep

    def split_stratified(self,keep_percentage=0.8):
        n=len(self.y)

        classes=max(self.y)+1
        d=dict(self.meta)
        d['x']=self.x
        d['y']=self.y
        test_dict={}
        train_dict={}
        train_indices=[]
        test_indices=[]
        y_np=np.array(self.y)
        for c in range(classes):
            indices=np.where(y_np==c)[0]
            #print(indices)
            random.shuffle(indices)
            n_class=len(indices)
            train_n=math.floor(n_class*keep_percentage)
            train_n=min(n_class-1,train_n)
            train_indices.extend(indices[:train_n])
            test_indices.extend(indices[train_n:])
            # train_x.extend(self.x[indices[]])
            # test_x.extend( self.x[indices[train_n:]])
            # train_y.extend(self.y[indices[:train_n]])
            # test_y.extend( self.y[indices[train_n:]])
        for key in d:
            if len(d[key])==len(self.y):
              train_list=train_dict.get(key,[])
              train_list.extend([d[key][i] for i in train_indices])
              train_dict[key]=train_list
              test_list=test_dict.get(key,[])
              test_list.extend([d[key][i] for i in test_indices])
              test_dict[key]=test_list
            else:
              train_dict[key]=d[key]
              test_dict[key]=d[key]
        #print(train_indices)
        train_x=train_dict.pop("x")
        train_y=train_dict.pop("y")
        test_x=test_dict.pop("x")
        test_y=test_dict.pop("y")
        # print(len(test_x))
        # print(test_x[0].shape)
        # train_x=np.concatenate(train_x,axis=0)
        # test_x =np.concatenate(test_x,axis=0)
        # train_y=np.concatenate(train_y,axis=0)
        # test_y =np.concatenate(test_y,axis=0)

        return (Dataset(self.id,train_x,train_y,train_dict),Dataset(self.id,test_x,test_y,test_dict))
