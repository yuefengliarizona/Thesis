import numpy as np
import torch
List1 =np.array([[ 4, 45,  8,  4],
       [ 2, 23,  6,  4]])

List2=np.array([ 2, 54, 13, 15])

similarity_scores = List1.dot(List2)/ (np.linalg.norm(List1, axis=1) * np.linalg.norm(List2))



print()

