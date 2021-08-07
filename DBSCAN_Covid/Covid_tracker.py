
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Set

from dataclasses import dataclass, field

@dataclass
class Covid_tracker:

    _data: pd.DataFrame = field(init = True)
    _dbmod: DBSCAN = DBSCAN(eps=0.0008288, 
                            min_samples=2,
                            metric='haversine')

    _clusters: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = pd.concat([data, self._data])

    def tracking(self, targ:str=None) -> None:

        data = self._data.copy()
        self._dbmod.fit(self._data[["Latitude", "Longitude"]])
        data['Cluster'] = self._dbmod.labels_.tolist()

        if targ not in data['User'].unique():
            print(f"Không tìm thấy đối tượng {targ} trong dữ liệu")

        elif targ in self._clusters:
            print(self._clusters[targ])
        else:
            contacts = set()
            targ_clusters = set(data[data['User']==targ]['Cluster'])

            for i in targ_clusters:
                if i != -1:
                    contacts.update(set(data[data['Cluster']==i]['User']))
                    contacts.remove(targ)

            self._clusters[targ] = contacts
            print(contacts)

    def visualize(self):

        data =self._data.copy()
        data['Cluster'] = self._dbmod.labels_.tolist()
        
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize = (7,7))
        
        g = sns.scatterplot(x=data['Longitude'], 
                        y=data['Latitude'], 
                        hue=data['Cluster'].astype(str))
        
        g.legend_.remove()

        plt.show(fig)