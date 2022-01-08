from haar_features import haar_like_features
import numpy as np


img = np.random.randint(0, 255, size=(24, 24))
ii = hf_functions.integral_image(img)

