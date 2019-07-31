# simple
sample_df = df.sample(100)

# stratified
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.25)
                                                   
# reservoir
import random

def generator(max):
    number = 1
    while number < max:
        number += 1
        yield number

# Create as stream generator
stream = generator(10000)

# Doing Reservoir Sampling from the stream
k=5
reservoir = []
for i, element in enumerate(stream):
    if i+1<= k:
        reservoir.append(element)
    else:
        probability = k/(i+1)
        if random.random() < probability:
            # Select item in stream and remove one of the k items already selected
             reservoir[random.choice(range(0,k))] = element
             
             
 
